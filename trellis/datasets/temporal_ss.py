import os
import glob
import zipfile
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from .sparse_structure_latent import SparseStructureLatentVisMixin


# Errors that indicate a corrupt / unreadable on-disk asset (truncated npz,
# half-written png, etc.). Used by the loader to skip & fall back to another
# scene instead of killing the DataLoader worker.
_CORRUPT_FILE_ERRORS = (
    zipfile.BadZipFile,
    OSError,
    EOFError,
    ValueError,
    UnidentifiedImageError,
)


class TemporalSS(SparseStructureLatentVisMixin, Dataset):
    """
    Multi-scene sequential frame dataset for temporal SS flow matching training.
    Each ``__getitem__`` returns ALL frames from one scene as a sequence.
    Designed for ``batch_size=1`` (one scene per step).

    Train scenes are discovered from ``<root>/{encode_done, ss_latents,
    renders_cond}``. Validation scenes live in a physically separate
    ``<root>/val/{ss_latents, renders_cond}`` subtree and are NOT included
    in training. Exposed via ``get_val_data()`` (first val scene) and
    ``get_all_val_scenes()`` (iterate all).

    Scenes shorter than ``window_size`` frames are silently dropped — the
    trainer slices a ``W``-frame window per step and requires at least
    ``W`` frames per scene. Mirrors the implicit drop the SLat dataset
    performs when it enumerates ``(scene, start_idx)`` pairs at init.

    Expected directory structure:
        root/
            encode_done/<scene_id>.done
            ss_latents/<scene_id>/frame_XXXX.npz   (mean: [8, 16, 16, 16])
            renders_cond/<scene_id>/view_YY/frame_XXXX.png  (512x512 RGBA)
            val/
                ss_latents/<scene_id>/frame_XXXX.npz
                renders_cond/<scene_id>/view_YY/frame_XXXX.png
        where ``view_00`` is always the canonical front view, and
        ``view_01..view_{N-1}`` are Hammersley-sampled alternatives. ONE
        view is chosen per ``__getitem__`` call and used for every frame
        of the returned clip.
    """
    def __init__(
        self,
        roots: str,
        *,
        image_size: int = 518,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        window_size: int = 3,
        val_num_scenes: int = 5,
        val_num_frames: Optional[int] = None,
        num_cond_views: int = 6,
        front_view_prob: float = 0.4,
        **kwargs,
    ):
        self.normalization = normalization
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
        super().__init__(
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        self.root = roots.split(',')[0]
        self.val_root = os.path.join(self.root, 'val')
        self.image_size = image_size
        self.value_range = (0, 1)
        assert int(window_size) >= 1, f'window_size must be >= 1, got {window_size}'
        self.window_size = int(window_size)
        self.val_num_frames = val_num_frames
        assert num_cond_views >= 1, f'num_cond_views must be >=1: {num_cond_views}'
        assert 0.0 <= front_view_prob <= 1.0, f'front_view_prob out of range: {front_view_prob}'
        self.num_cond_views = num_cond_views
        self.front_view_prob = front_view_prob

        # ---- Train scenes from <root>/{encode_done, ss_latents, renders_cond} ----
        # Val samples were physically moved to <root>/val/, so their ss_latents
        # subdirs no longer exist under <root>/ and they are naturally excluded.
        encode_done_dir = os.path.join(self.root, 'encode_done')
        scene_ids = sorted([
            f.replace('.done', '') for f in os.listdir(encode_done_dir) if f.endswith('.done')
        ])
        self.scenes = []
        skipped_incomplete = []
        skipped_too_short = []
        for sid in scene_ids:
            ss_dir = os.path.join(self.root, 'ss_latents', sid)
            cond_dir = os.path.join(self.root, 'renders_cond', sid)
            if not (os.path.isdir(ss_dir) and os.path.isdir(cond_dir)):
                continue
            num_frames = len(glob.glob(os.path.join(ss_dir, 'frame_*.npz')))
            # Scenes shorter than the training window are skipped — the
            # trainer slices a window of `self.window_size` frames per step.
            if num_frames < self.window_size:
                if num_frames > 0:
                    skipped_too_short.append((sid, num_frames))
                continue
            # Require every view dir to exist AND to contain at least
            # `num_frames` `frame_XXXX.png` files — catches partially-rendered
            # scenes where the manifest says "done" but some PNGs are missing.
            missing_views = []
            for i in range(self.num_cond_views):
                vdir = os.path.join(cond_dir, f'view_{i:02d}')
                if not os.path.isdir(vdir):
                    missing_views.append(f'view_{i:02d}')
                    continue
                png_count = sum(
                    1 for e in os.listdir(vdir)
                    if e.startswith('frame_') and e.endswith('.png')
                )
                if png_count < num_frames:
                    missing_views.append(f'view_{i:02d}({png_count}/{num_frames})')
            if missing_views:
                skipped_incomplete.append((sid, missing_views))
                continue
            self.scenes.append((sid, num_frames))
        if skipped_incomplete:
            print(f"TemporalSS: skipped {len(skipped_incomplete)} "
                  f"scene(s) with incomplete cond views (num_cond_views={self.num_cond_views})")
            for sid, mv in skipped_incomplete[:5]:
                print(f"  skip: {sid} missing {mv}")
        if skipped_too_short:
            print(f"TemporalSS: skipped {len(skipped_too_short)} "
                  f"scene(s) with <window_size={self.window_size} frames")
            for sid, nf in skipped_too_short[:5]:
                print(f"  skip: {sid} ({nf} frames)")

        # ---- Val scenes from <root>/val/{ss_latents, renders_cond} ----
        # Cap to val_num_scenes scenes and val_num_frames frames per scene.
        # If either cap exceeds what's available, use everything available.
        val_scenes_all = []
        if os.path.isdir(self.val_root):
            val_ss = os.path.join(self.val_root, 'ss_latents')
            val_cond = os.path.join(self.val_root, 'renders_cond')
            if os.path.isdir(val_ss) and os.path.isdir(val_cond):
                for sid in sorted(os.listdir(val_ss)):
                    ss_dir = os.path.join(val_ss, sid)
                    cond_dir = os.path.join(val_cond, sid)
                    if not (os.path.isdir(ss_dir) and os.path.isdir(cond_dir)):
                        continue
                    nf = len(glob.glob(os.path.join(ss_dir, 'frame_*.npz')))
                    if nf == 0:
                        continue
                    if val_num_frames is not None:
                        nf = min(nf, val_num_frames)
                    val_scenes_all.append((sid, nf))
        if val_num_scenes > 0:
            self.val_scenes = val_scenes_all[:val_num_scenes]
        else:
            self.val_scenes = val_scenes_all

        print(f"TemporalSS: {len(self.scenes)} train / "
              f"{len(self.val_scenes)} val scenes "
              f"(window_size={self.window_size}, "
              f"val cap: val_num_scenes={val_num_scenes}, "
              f"val_num_frames={val_num_frames}), "
              f"num_cond_views={self.num_cond_views} front_view_prob={self.front_view_prob}")
        for sid, nf in self.val_scenes:
            print(f"  val: {sid} ({nf} frames)")

    def __len__(self):
        return len(self.scenes)

    def _pick_view(self):
        """Sample a cond view index: front (view_00) with front_view_prob,
        otherwise uniform among the non-front views."""
        if self.num_cond_views <= 1:
            return 0
        if np.random.random() < self.front_view_prob:
            return 0
        return 1 + np.random.randint(self.num_cond_views - 1)

    def _load_frame(self, scene_id, frame_idx, view_idx=0, from_val=False):
        """from_val=True reads from <root>/val/, otherwise from <root>/."""
        base = self.val_root if from_val else self.root
        # Load SS latent
        ss_path = os.path.join(base, 'ss_latents', scene_id, f'frame_{frame_idx:04d}.npz')
        with np.load(ss_path) as data:
            z = torch.tensor(data['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        # Load conditioning image at (view_idx, frame_idx).
        img_path = os.path.join(
            base, 'renders_cond', scene_id,
            f'view_{view_idx:02d}', f'frame_{frame_idx:04d}.png',
        )
        with Image.open(img_path) as src:
            src.load()
            resized = src.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = resized.getchannel(3)
        rgb = resized.convert('RGB')
        image = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)

        return z, image

    def _try_load_scene(self, index, view_idx):
        """Load all frames of a scene. Returns None if any frame is corrupt
        (caller should skip the scene); raises on unexpected errors."""
        scene_id, num_frames = self.scenes[index]
        latents = []
        images = []
        for i in range(num_frames):
            try:
                z, img = self._load_frame(scene_id, i, view_idx=view_idx)
            except _CORRUPT_FILE_ERRORS as e:
                print(f'[Dataset] Corrupt asset, skipping scene: '
                      f'scene={scene_id} frame={i} view={view_idx} err={type(e).__name__}: {e}',
                      flush=True)
                return None
            latents.append(z)
            images.append(img)
        return scene_id, num_frames, latents, images

    def __getitem__(self, index):
        # Sample ONE cond view for the entire clip so temporal conditioning
        # stays consistent across frames.
        view_idx = self._pick_view()

        # If the chosen scene has any corrupt asset, fall back to other scenes.
        # Try in order: the requested index, then a bounded random walk so
        # different workers don't dogpile the same fallback scene.
        n = len(self.scenes)
        max_retries = min(16, n)
        tried = set()
        cur = index
        for _ in range(max_retries):
            if cur in tried:
                cur = int(np.random.randint(n))
                continue
            tried.add(cur)
            result = self._try_load_scene(cur, view_idx)
            if result is not None:
                scene_id, num_frames, latents, images = result
                return {
                    'x_0': torch.stack(latents),    # (T, 8, 16, 16, 16)
                    'cond': torch.stack(images),     # (T, 3, 518, 518)
                    'num_frames': num_frames,
                }
            cur = int(np.random.randint(n))

        raise RuntimeError(
            f'[Dataset] Could not load any usable scene after {max_retries} retries '
            f'(starting index={index}). Check dataset integrity.'
        )

    def _load_val_scene(self, sid: str, from_val: bool = True):
        """Load all frames of a single val scene. Always uses view_00."""
        base = self.val_root if from_val else self.root
        ss_dir = os.path.join(base, 'ss_latents', sid)
        num_frames = len(glob.glob(os.path.join(ss_dir, 'frame_*.npz')))
        if self.val_num_frames is not None:
            num_frames = min(num_frames, self.val_num_frames)

        latents = []
        images = []
        for i in range(num_frames):
            z, img = self._load_frame(sid, i, view_idx=0, from_val=from_val)
            latents.append(z)
            images.append(img)

        return {
            'scene_id': sid,
            'x_0': torch.stack(latents),
            'cond': torch.stack(images),
            'num_frames': num_frames,
        }

    def get_val_data(self):
        """Get first validation scene data for snapshot rendering."""
        if self.val_scenes:
            sid, _ = self.val_scenes[0]
            return self._load_val_scene(sid, from_val=True)
        sid, _ = self.scenes[0]
        return self._load_val_scene(sid, from_val=False)

    def get_all_val_scenes(self):
        """Iterator over all validation scenes for full val MSE computation."""
        if self.val_scenes:
            for sid, _ in self.val_scenes:
                yield self._load_val_scene(sid, from_val=True)
        else:
            sid, _ = self.scenes[0]
            yield self._load_val_scene(sid, from_val=False)

    def __str__(self):
        first_val = self.val_scenes[0][0] if self.val_scenes else (self.scenes[0][0] if self.scenes else 'N/A')
        return (
            f"TemporalSS\n"
            f"  - Train scenes: {len(self.scenes)}\n"
            f"  - Val scenes: {len(self.val_scenes)} (first: {first_val})\n"
            f"  - Image size: {self.image_size}\n"
            f"  - Normalization: {self.normalization is not None}\n"
            f"  - Cond views: {self.num_cond_views} (front_view_prob={self.front_view_prob})"
        )