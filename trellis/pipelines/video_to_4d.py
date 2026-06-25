"""Combined SS + SLat Diffusion-Forcing video-to-4D pipeline.

Per scene the pipeline:

  1. Encode each frame's RGB(A) cond image with DINOv2.
  2. Sample the sparse-structure (SS) voxel for each frame autoregressively
     with a sliding KV cache.
  3. Decode each SS sample -> continuous occupancy -> threshold -> coords.
  4. Sample SLat features over those coords autoregressively.
  5. Decode SLat to mesh + gaussian and render / save.

Only ``ar_kv`` (streaming AR with a sliding KV cache) is supported. Both SS and
SLat use :class:`FlowEulerKVCacheGuidanceIntervalSampler` with **dual** KV
caches: a positive cache built with ``cond`` and a negative cache built with
``neg_cond``, so classifier-free guidance runs the entire negative path —
current frame and cached context — unconditionally (full-window-drop CFG), with
the standard ``(1+cfg)·v_pos − cfg·v_neg`` formula gated to ``cfg_interval`` on
the ``rescale_t`` schedule.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..modules.sparse.basic import SparseTensor
from .samplers import FlowEulerKVCacheGuidanceIntervalSampler
from ..utils.video_to_4d_utils import (
    load_diffforcing_model,
    DinoImageEncoder,
    load_slat_decoder,
    load_mesh_decoder,
    make_one_camera,
    render_gaussian_view,
    ShiftingBank,
    DEFAULT_YAW0,
    DEFAULT_PITCH,
    DEFAULT_DISTANCE,
)


class VideoTo4DPipeline:
    """Diffusion Forcing SS + SLat → mesh + gaussian, autoregressive over time."""

    def __init__(
        self,
        ss_model: torch.nn.Module,
        ss_decoder: torch.nn.Module,
        slat_model: torch.nn.Module,
        slat_dec_gauss: torch.nn.Module,
        slat_dec_mesh: torch.nn.Module,
        slat_normalization: Optional[dict],
        ss_window_size: int,
        slat_window_size: int,
        slat_sigma_min: float,
        dino_encoder: DinoImageEncoder,
        kv_sampler: FlowEulerKVCacheGuidanceIntervalSampler,
        device: str = 'cuda',
        rescale_t: float = 1.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.ss_model = ss_model
        self.ss_decoder = ss_decoder
        self.slat_model = slat_model
        self.slat_dec_gauss = slat_dec_gauss
        self.slat_dec_mesh = slat_dec_mesh
        self.slat_normalization = slat_normalization
        self.ss_window_size = ss_window_size
        self.slat_window_size = slat_window_size
        self.slat_sigma_min = slat_sigma_min
        self.dino = dino_encoder
        # One dual-cache sampler drives both stages (dense SS + sparse SLat).
        self.kv_sampler = kv_sampler
        self.device = device
        # Euler schedule controls (naive defaults: linear schedule, CFG every
        # step). Honored by both SS and SLat sampling.
        self.rescale_t = rescale_t
        self.cfg_interval = tuple(cfg_interval)

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def from_ckpts(
        cls,
        ss_config: str,
        ss_weights: str,
        slat_config: str,
        slat_weights: str,
        rescale_t: float = 1.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
    ):
        """Load SS + SLat denoisers and build the end-to-end video-to-4D pipeline.

        ``*_config`` are the architecture JSONs that ship with this repo
        (``config/temporal_{ss,slat}_flow_dit_w3_diffforcing.json``); ``*_weights``
        are the `.pt` files from the MORPHOS release (the HF repo ships only
        `.pt`, no paired JSON). Decoder paths (SS coord scaffold + SLat Gaussian)
        are read from the ``dataset.args`` block of each config (``pretrained_ss_dec``
        / ``pretrained_slat_dec``)."""
        from .. import models
        ss_model, ss_cfg = load_diffforcing_model(ss_config, ss_weights)
        slat_model, slat_cfg = load_diffforcing_model(slat_config, slat_weights)

        # Decoders (frozen, pretrained). Paths come from the symmetric
        # `pretrained_*_dec` args in the dataset block of each config.
        ss_dec_path = ss_cfg['dataset']['args'].get(
            'pretrained_ss_dec',
            'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        )
        ss_decoder = models.from_pretrained(ss_dec_path).cuda().eval()
        print(f'[load] SS decoder from {ss_dec_path}')

        slat_dec_path = slat_cfg['dataset']['args'].get(
            'pretrained_slat_dec',
            'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        )
        slat_dec_gauss = load_slat_decoder(slat_dec_path)
        slat_dec_mesh = load_mesh_decoder()

        slat_normalization = slat_cfg['dataset']['args'].get('normalization')
        ss_window_size = ss_cfg['models']['denoiser']['args'].get('window_size', 3)
        # SLat: prefer trainer.window_size (canonical), fall back to legacy
        # `prev_window` (= window_size - 1) for older configs.
        slat_train_args = slat_cfg['trainer']['args']
        if 'window_size' in slat_train_args:
            slat_window_size = int(slat_train_args['window_size'])
        else:
            slat_window_size = int(slat_train_args.get('prev_window', 2)) + 1
        slat_sigma_min = slat_train_args.get('sigma_min', 1e-5)

        # One generic dual-cache sampler for both stages. It calls
        # forward_with_kvcache directly and auto-detects dense (SS) vs sparse
        # (SLat) latents; sigma_min is unused by its velocity-Euler step.
        kv_sampler = FlowEulerKVCacheGuidanceIntervalSampler(sigma_min=slat_sigma_min)
        return cls(
            ss_model=ss_model,
            ss_decoder=ss_decoder,
            slat_model=slat_model,
            slat_dec_gauss=slat_dec_gauss,
            slat_dec_mesh=slat_dec_mesh,
            slat_normalization=slat_normalization,
            ss_window_size=ss_window_size,
            slat_window_size=slat_window_size,
            slat_sigma_min=slat_sigma_min,
            dino_encoder=DinoImageEncoder(),
            kv_sampler=kv_sampler,
            rescale_t=rescale_t,
            cfg_interval=cfg_interval,
        )

    # -----------------------------------------------------------------
    # Cond encoding
    # -----------------------------------------------------------------
    @torch.no_grad()
    def encode_cond_video(self, cond_video: torch.Tensor) -> List[torch.Tensor]:
        """cond_video [T, 3, H, W] → list of per-frame DINOv2 features (T items)."""
        return [self.dino(cond_video[i].unsqueeze(0).cuda()) for i in range(cond_video.shape[0])]

    # -----------------------------------------------------------------
    # Stage 1: per-frame SS sampling (ar_kv, dual cond/uncond KV cache)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_ss_sequence(
        self,
        cond_per_frame: List[torch.Tensor],
        steps: int = 25,
        cfg_strength: float = 3.0,
    ) -> List[torch.Tensor]:
        """Run SS Diffusion Forcing autoregressively with a sliding KV cache.
        Returns list of (1, 8, 16, 16, 16) dense tensors, one per frame.

        Dual-cache full-window CFG: a positive cache built with ``cond`` and a
        negative cache built with ``neg_cond``; the negative branch denoises
        with ``neg_cond`` against the uncond cache.
        """
        T = len(cond_per_frame)
        ss_model = self.ss_model
        in_ch = ss_model.base_model.in_channels
        reso = ss_model.base_model.resolution
        W = self.ss_window_size

        pos_bank = ShiftingBank(max_window=max(0, W - 1))
        neg_bank = ShiftingBank(max_window=max(0, W - 1))
        samples: List[torch.Tensor] = []

        for fidx in range(T):
            cond = cond_per_frame[fidx]
            neg_cond = torch.zeros_like(cond)
            pos_window = pos_bank.window()
            neg_window = neg_bank.window()

            noise = torch.randn(1, in_ch, reso, reso, reso, device=self.device)
            res = self.kv_sampler.sample(
                ss_model, noise=noise, cond=cond, neg_cond=neg_cond, frame_idx=fidx,
                pos_kv_caches=pos_window, neg_kv_caches=neg_window,
                steps=steps, rescale_t=self.rescale_t,
                cfg_strength=cfg_strength, cfg_interval=self.cfg_interval,
                verbose=False,
            )
            x_0_pred = res.samples
            samples.append(x_0_pred)

            # Build BOTH caches from the denoised frame (kwarg now unified
            # with SLat: prev_kv_caches=).
            pos_cache = ss_model.build_kv_cache(
                x=x_0_pred, frame_idx=fidx, cond=cond, t_noise=0.0,
                prev_kv_caches=pos_window,
            )
            neg_cache = ss_model.build_kv_cache(
                x=x_0_pred, frame_idx=fidx, cond=neg_cond, t_noise=0.0,
                prev_kv_caches=neg_window,
            )
            pos_bank.append(pos_cache)
            neg_bank.append(neg_cache)

        return samples

    @torch.no_grad()
    def ss_to_coords(self, ss_dense: torch.Tensor, batch_idx: int = 0,
                     threshold: float = 0.0) -> torch.Tensor:
        """Decode (1, 8, 16, 16, 16) SS latent → coords [N, 4] for SparseTensor.

        SS coord-scaffold decoding: ``coords = argwhere(decoder(z) > threshold)``,
        projected from ``[N, 5]`` (batch, c, x, y, z) to ``[N, 4]`` for the
        ``(batch_idx, x, y, z)`` SparseTensor coord format.
        """
        decoded = self.ss_decoder(ss_dense)
        idx = torch.argwhere(decoded > threshold)[:, [0, 2, 3, 4]].int()
        if batch_idx != 0:
            idx[:, 0] = batch_idx
        return idx

    # -----------------------------------------------------------------
    # Stage 2: per-frame SLat sampling (ar_kv, dual cond/uncond KV cache)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_slat_sequence(
        self,
        ss_coords_per_frame: List[torch.Tensor],
        cond_per_frame: List[torch.Tensor],
        steps: int = 25,
        cfg_strength: float = 3.0,
    ) -> List[SparseTensor]:
        """Run SLat Diffusion Forcing autoregressively (ar_kv) over the
        SS-derived coord scaffold, with dual cond/uncond KV caches."""
        model = self.slat_model
        T = len(ss_coords_per_frame)
        W = self.slat_window_size
        sigma_min = self.slat_sigma_min
        in_ch = model.base_model.in_channels if hasattr(model, 'base_model') else 8

        pos_bank = ShiftingBank(max_window=max(0, W - 1))
        neg_bank = ShiftingBank(max_window=max(0, W - 1))
        sample_results: List[SparseTensor] = []

        for fidx in range(T):
            cond = cond_per_frame[fidx]
            neg_cond = torch.zeros_like(cond)
            coords = ss_coords_per_frame[fidx].to(self.device)

            pos_window = pos_bank.window()
            neg_window = neg_bank.window()

            init_feats = torch.randn(coords.shape[0], in_ch, device=self.device)
            x_t = SparseTensor(coords=coords, feats=init_feats)
            res = self.kv_sampler.sample(
                model, noise=x_t, cond=cond, neg_cond=neg_cond, frame_idx=fidx,
                pos_kv_caches=pos_window, neg_kv_caches=neg_window,
                steps=steps, rescale_t=self.rescale_t,
                cfg_strength=cfg_strength, cfg_interval=self.cfg_interval,
                verbose=False,
            )
            denoised = res.samples
            sample_results.append(denoised)

            # Build BOTH caches from the denoised frame (SLat uses prev_kv_caches=).
            pos_cache = model.build_kv_cache(
                denoised, t_noise=0.0, frame_idx=fidx, cond=cond,
                sigma_min=sigma_min, prev_kv_caches=pos_window,
            )
            neg_cache = model.build_kv_cache(
                denoised, t_noise=0.0, frame_idx=fidx, cond=neg_cond,
                sigma_min=sigma_min, prev_kv_caches=neg_window,
            )
            pos_bank.append(pos_cache)
            neg_bank.append(neg_cache)

        return sample_results

    # -----------------------------------------------------------------
    # Gaussian decoding
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _denorm_slat(self, slat: SparseTensor) -> SparseTensor:
        """Apply training-time per-channel denormalization (no decode).
        Returns ``slat`` unchanged when ``slat_normalization is None``."""
        if self.slat_normalization is None:
            return slat
        mean = torch.tensor(self.slat_normalization['mean']).reshape(1, -1).cuda()
        std = torch.tensor(self.slat_normalization['std']).reshape(1, -1).cuda()
        return slat.replace(slat.feats * std + mean)

    @torch.no_grad()
    def _decode_one_to_gaussian(self, slat: SparseTensor):
        """De-normalize feats then decode SLat → Gaussian rep (single-frame)."""
        return self.slat_dec_gauss(self._denorm_slat(slat))[0]

    # -----------------------------------------------------------------
    # Decode + render + save
    # -----------------------------------------------------------------
    @torch.no_grad()
    def decode_and_save(
        self,
        slat_per_frame: List[SparseTensor],
        scene_id: str,
        mode: str,
        output_dir: str,
        dataset_name: str,
        save_mesh: bool = True,
        save_slat: bool = True,
        textured: bool = False,
        mesh_simplify: float = 0.95,
        mesh_texture_size: int = 1024,
        render_resolution: int = 512,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        appearance_yaw_offset_deg: float = 0.0,
    ):
        """Decode each frame's SLat into the eval-ready per-frame tree:

            {output_dir}/{dataset}/{mode}/appearance/{scene_id}/{0|90|180|270}/frame_NNNN.png
            {output_dir}/{dataset}/{mode}/geometry/{scene_id}/frame_NNNN.glb
            {output_dir}/{dataset}/{mode}/slat/{scene_id}/frame_NNNN.npz

        - Appearance: per-azimuth per-frame PNGs rendered from the default pose
          (yaw/pitch/distance). View 0 is at the default yaw; views 1/2/3 at
          yaw + π/2, π, 3π/2. The four directories are labelled by the RELATIVE
          azimuth offset (0, 90, 180, 270 deg) — matching the GT view ordering
          used by ``trellis.evaluation.metric_appearance``. The camera is fixed
          across the whole video.
        - Geometry: per-frame .glb (geometry-only, or textured if requested).
        - SLat: per-frame .npz with the DENORMALIZED SparseTensor
          (``coords [N,3] int32`` after dropping the batch slot, ``feats [N,C]
          float32``).
        """
        from PIL import Image

        mode_root = os.path.join(output_dir, dataset_name, mode)
        geometry_dir = os.path.join(mode_root, 'geometry', scene_id)
        appearance_dir = os.path.join(mode_root, 'appearance', scene_id)
        slat_dir = os.path.join(mode_root, 'slat', scene_id)

        # Azimuth directories: relative offsets from the default yaw0.
        azimuth_dirs: Dict[int, str] = {}
        for az in (0, 90, 180, 270):
            d = os.path.join(appearance_dir, str(az))
            os.makedirs(d, exist_ok=True)
            azimuth_dirs[az] = d
        if save_mesh:
            os.makedirs(geometry_dir, exist_ok=True)
        if save_slat:
            os.makedirs(slat_dir, exist_ok=True)

        # 1) Fixed default camera pose for the whole video.
        yaw0, pitch, distance = DEFAULT_YAW0, DEFAULT_PITCH, DEFAULT_DISTANCE

        # 2) Per-azimuth gaussian PNGs. View k uses
        #    yaw = yaw0 + appearance_yaw_offset + k*pi/2; the relative-offset
        #    label (0, 90, 180, 270) goes in the path. appearance_yaw_offset_deg
        #    rotates the rendered content under the labels without moving them
        #    (e.g. 180 -> label 0 shows what label 180 used to).
        az_for_view = {0: 0, 1: 90, 2: 180, 3: 270}
        yaw_offset = np.deg2rad(appearance_yaw_offset_deg)
        for k in range(4):
            yaw_k = yaw0 + yaw_offset + k * (np.pi / 2)
            ext, intr = make_one_camera(yaw_k, pitch, distance)
            for i, slat in enumerate(slat_per_frame):
                gauss = self._decode_one_to_gaussian(slat)
                color = render_gaussian_view(
                    gauss, ext, intr, distance=distance,
                    resolution=render_resolution, bg_color=bg_color,
                )
                arr = (color.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                out_path = os.path.join(
                    azimuth_dirs[az_for_view[k]], f'frame_{i:04d}.png',
                )
                Image.fromarray(arr).save(out_path)
        print(f'  saved {4 * len(slat_per_frame)} appearance frames to {appearance_dir}')

        # 3) Per-frame .glb (geometry-only or textured) + per-frame SLat .npz.
        if save_mesh or save_slat:
            import trimesh
            n_saved_mesh = 0
            n_saved_slat = 0
            for i, slat in enumerate(slat_per_frame):
                slat_dec = self._denorm_slat(slat)

                if save_slat:
                    # Drop the SparseTensor batch column (slot 0) so the on-disk
                    # layout matches dataset_slat_dynamic_v60_c6_f12/slat_latents/.
                    coords_np = slat_dec.coords[:, 1:].detach().cpu().numpy().astype(np.int32)
                    feats_np = slat_dec.feats.detach().cpu().numpy().astype(np.float32)
                    np.savez(
                        os.path.join(slat_dir, f'frame_{i:04d}.npz'),
                        coords=coords_np, feats=feats_np,
                    )
                    n_saved_slat += 1

                if save_mesh:
                    mesh = self.slat_dec_mesh(slat_dec)[0]
                    if not getattr(mesh, 'success', True) or mesh.vertices.shape[0] == 0:
                        print(f'  [warn] {scene_id} frame {i}: mesh extraction failed; skipping')
                        continue
                    out_path = os.path.join(geometry_dir, f'frame_{i:04d}.glb')
                    if textured:
                        from trellis.utils import postprocessing_utils
                        gauss = self.slat_dec_gauss(slat_dec)[0]
                        glb = postprocessing_utils.to_glb(
                            gauss, mesh, simplify=mesh_simplify,
                            fill_holes=True, texture_size=mesh_texture_size,
                        )
                        glb.export(out_path)
                    else:
                        verts = mesh.vertices.detach().cpu().numpy()
                        faces = mesh.faces.detach().cpu().numpy()
                        trimesh.Trimesh(
                            vertices=verts, faces=faces, process=False,
                        ).export(out_path)
                    n_saved_mesh += 1

            if save_mesh:
                print(f'  saved {n_saved_mesh}/{len(slat_per_frame)} mesh frames to {geometry_dir}')
            if save_slat:
                print(f'  saved {n_saved_slat}/{len(slat_per_frame)} slat frames to {slat_dir}')
