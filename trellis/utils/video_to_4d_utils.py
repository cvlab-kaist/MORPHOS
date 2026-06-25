"""Helpers for the combined SS + SLat Diffusion-Forcing video-to-4D pipeline
(:class:`trellis.pipelines.VideoTo4DPipeline`).

Contents:
  - checkpoint loading for Diffusion-Forcing SS / SLat models;
  - a DINOv2 image-conditioning encoder (mirrors the stock TRELLIS pipeline);
  - SLat gaussian / mesh decoder loaders;
  - camera rig + per-view gaussian / mesh-normal rendering;
  - mp4 / cond-video / mesh-saving utilities;
  - a bounded sliding-window KV-cache bank.

Heavy package deps (models, renderers, postprocessing) are imported lazily so
this module is import-order-safe within ``trellis.pipelines``.
"""
import os
import json
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import utils3d

from ..modules.sparse.basic import SparseTensor

# Camera rig defaults. View 0 of the posed 4-view rig is the calibration
# target; near/far scale linearly with distance (distance=2 -> near=0.8,
# far=1.6), preserving the convention used across the SLat inference scripts.
DEFAULT_YAW0 = 0.0
DEFAULT_PITCH = float(np.pi / 6)
DEFAULT_DISTANCE = 2.5
NEAR_RATIO, FAR_RATIO = 0.4, 0.8


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_diffforcing_model(config_path: str, weights_path: str):
    """Load a Diffusion Forcing model (SS or SLat) for inference.

    Architecture comes from ``config_path`` (a JSON in ``config/`` shipped with
    this repo); the trained weights come from ``weights_path`` (a `.pt` file —
    the MORPHOS release ships only `.pt` per stage). Returns ``(model, cfg)``.

    ``pretrained_base`` from the config is dropped before instantiation: the
    `.pt` already carries the full trained weights, so re-downloading the
    base checkpoint would be wasted I/O immediately overwritten by
    ``load_state_dict``.
    """
    from .. import models
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f'Weights not found: {weights_path}')
    with open(config_path) as f:
        cfg = json.load(f)
    model_cfg = cfg['models']['denoiser']
    init_args = {k: v for k, v in model_cfg['args'].items() if k != 'pretrained_base'}
    model = getattr(models, model_cfg['name'])(**init_args, pretrained_base=None)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if model_cfg['args'].get('use_fp16', False):
        model.convert_to_fp16()
    model.cuda().eval()
    print(f'[load] {model_cfg["name"]}: weights={weights_path} config={config_path}')
    return model, cfg


def load_slat_decoder(name='microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16'):
    from .. import models
    dec = models.from_pretrained(name).cuda().eval()
    print(f'[load] SLat gaussian decoder: {name}')
    return dec


def load_mesh_decoder(name='microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16'):
    from .. import models
    dec = models.from_pretrained(name).cuda().eval()
    print(f'[load] SLat mesh decoder: {name}')
    return dec


# ============================================================================
# DINOv2 image-conditioning encoder
# ============================================================================

class DinoImageEncoder:
    """DINOv2 image-conditioning encoder.

    The torch.hub ``dinov2_vitl14_reg`` backbone, ImageNet normalization,
    ``x_prenorm`` patch tokens, and a final token-wise layer-norm.

    Mirrors the **training** encoder
    (``ImageConditionedMixin.encode_image``): the DINOv2 forward runs under
    **bf16 autocast**, then features are cast back to fp32 before the
    layer-norm. This matches the conditioning distribution the Diffusion-
    Forcing models were trained on (train/inference parity) and lets the bf16
    fa2F attention kernel dispatch on B200/sm_100.
    """
    def __init__(self, model_name: str = 'dinov2_vitl14_reg', device: str = 'cuda'):
        from torchvision import transforms
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval().to(device)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W) in [0, 1]
        x = self.transform(image).to(self.device)
        # bf16 autocast around the backbone (matches the trainer); cast back to
        # fp32 before the layer-norm, which the trainer also does in fp32.
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = self.model(x, is_training=True)['x_prenorm']
        feats = feats.float()
        return F.layer_norm(feats, feats.shape[-1:])


# ============================================================================
# Camera rig + per-view rendering
# ============================================================================

def make_one_camera(yaw: float, pitch: float, distance: float):
    """Single camera: pose at yaw / pitch / distance, looking at origin."""
    orig = torch.tensor([
        np.sin(yaw) * np.cos(pitch),
        np.cos(yaw) * np.cos(pitch),
        np.sin(pitch),
    ]).float().cuda() * float(distance)
    fov = torch.deg2rad(torch.tensor(40.0)).cuda()
    ext = utils3d.torch.extrinsics_look_at(
        orig,
        torch.tensor([0.0, 0.0, 0.0]).float().cuda(),
        torch.tensor([0.0, 0.0, 1.0]).float().cuda(),
    )
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return ext, intr


def make_4view_cameras_posed(yaw0: float, pitch: float, distance: float):
    """Posed 4-view rig: view 0 at (yaw0, pitch, distance); views 1-3 at the
    same pitch / distance with yaws yaw0 + {π/2, π, 3π/2}."""
    yaws = [yaw0, yaw0 + np.pi / 2, yaw0 + np.pi, yaw0 + 3 * np.pi / 2]
    exts, ints = [], []
    for y in yaws:
        ext, intr = make_one_camera(y, pitch, distance)
        exts.append(ext)
        ints.append(intr)
    return exts, ints


def make_4view_cameras():
    """Canonical fixed 4-view rig (yaw {0, π/2, π, 3π/2}, pitch π/6, dist 2),
    used for the mesh-normal sanity video."""
    return make_4view_cameras_posed(0.0, float(np.pi / 6), 2.0)


def enumerate_range(start: float, end: float, gap: float) -> List[float]:
    """Inclusive arithmetic enumeration. `start==end` returns one item."""
    if gap <= 0 and start == end:
        return [float(start)]
    if gap <= 0:
        raise ValueError(f'gap must be > 0 for non-degenerate range; got {gap}')
    if end < start:
        raise ValueError(f'end ({end}) < start ({start})')
    out, d = [], float(start)
    while d <= float(end) + 1e-6:
        out.append(round(d, 6))
        d += float(gap)
    return out


@torch.no_grad()
def render_gaussian_view(
    rep,
    ext: torch.Tensor,
    intr: torch.Tensor,
    distance: float,
    resolution: int = 512,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Render a single view of a Gaussian rep. Returns (3, H, W) in [0,1] on
    cuda. near/far scale linearly with distance (distance=2 -> near=0.8,
    far=1.6)."""
    from ..renderers import GaussianRenderer
    renderer = GaussianRenderer({
        'resolution': int(resolution),
        'near': NEAR_RATIO * float(distance),
        'far':  FAR_RATIO * float(distance),
        'bg_color': tuple(float(x) for x in bg_color),
        'ssaa': 1,
    })
    out = renderer.render(rep, ext, intr)
    return out['color']


@torch.no_grad()
def render_4view_grid_mesh_normal(mesh_decoder, latents: List[SparseTensor],
                                  normalization: Optional[dict],
                                  tile_size: int = 256) -> List[np.ndarray]:
    """Decode each latent to a mesh and rasterize normal-shaded views from the
    canonical fixed 4-view rig, tiled into a 2x2 grid. Returns HWC uint8 frames.
    """
    from ..renderers.mesh_renderer import MeshRenderer

    exts, ints = make_4view_cameras()
    if normalization is not None:
        mean = torch.tensor(normalization['mean']).reshape(1, -1).cuda()
        std = torch.tensor(normalization['std']).reshape(1, -1).cuda()

    mesh_renderer = MeshRenderer(
        rendering_options={'resolution': tile_size, 'near': 1, 'far': 100, 'ssaa': 4}
    )

    frames = []
    for st in latents:
        st_dec = st
        if normalization is not None:
            st_dec = st.replace(st.feats * std + mean)
        mesh = mesh_decoder(st_dec)[0]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            tile = torch.zeros(3, tile_size * 2, tile_size * 2, device='cuda')
            frames.append((tile * 255).byte().permute(1, 2, 0).cpu().numpy())
            continue

        tile = torch.zeros(3, tile_size * 2, tile_size * 2, device='cuda')
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = mesh_renderer.render(mesh, ext, intr)
            normal = res['normal']  # (3, H, W) in [0, 1]
            if normal.shape[-1] != tile_size:
                normal = F.interpolate(normal.unsqueeze(0), size=(tile_size, tile_size),
                                       mode='bilinear', align_corners=False).squeeze(0)
            r, c = j // 2, j % 2
            tile[:, r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size] = normal

        frames.append((tile.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())
    return frames


# ============================================================================
# Video / mesh saving
# ============================================================================

def cond_to_video_frames(scene: dict, tile_size: int = 512) -> List[np.ndarray]:
    """Convert conditioning images (`scene['all_cond']`, [T,3,H,W] in [0,1]) to
    uint8 HWC frames."""
    frames = []
    for img in scene['all_cond']:
        x = img.clamp(0, 1)
        if x.shape[-1] != tile_size:
            x = F.interpolate(x.unsqueeze(0), size=(tile_size, tile_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        frames.append((x * 255).byte().permute(1, 2, 0).numpy())
    return frames


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f'  saved {path}')


# ============================================================================
# Sliding-window KV-cache bank
# ============================================================================

class ShiftingBank:
    """Bounded sliding window of the most-recent per-frame KV caches.

    `window()` returns up to `max_window` = W-1 caches (oldest-first) — the
    prev part of the model's full window; the current frame is appended
    separately by the sampler/model. Returns None when empty.
    """
    def __init__(self, max_window: int):
        self.max_window = max_window
        self.buf: List[Any] = []

    def append(self, cache):
        self.buf.append(cache)
        while self.max_window >= 0 and len(self.buf) > self.max_window:
            self.buf.pop(0)

    def window(self) -> Optional[list]:
        return list(self.buf) if self.buf else None
