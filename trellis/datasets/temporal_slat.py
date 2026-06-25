import os
import glob
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .structured_latent import SLatVisMixin


class TemporalSLat(SLatVisMixin, Dataset):
    """
    Multi-scene random-access temporal SLat dataset.

    Each __getitem__ returns ONE W-frame window of consecutive frames from
    a random scene. The window is a uniform random sample from all
    (scene, start_idx) pairs where start_idx + W <= num_frames; every frame
    in the returned window is treated equally by the trainer and the model.
    Scenes with fewer than W frames contribute no windows.

    Train scenes are discovered from `<root>/{encode_done, slat_latents,
    renders_cond}`. Validation scenes live in a physically separate
    `<root>/val/{slat_latents, renders_cond}` subtree and are NOT included
    in training. The dataset only indexes train scenes; val scenes are
    exposed via `get_val_data()` and `get_all_val_scenes()` for snapshot /
    val MSE.

    Expected directory structure:
        root/
            encode_done/<scene_id>.done
            slat_latents/<scene_id>/frame_XXXX.npz     (coords: [N,3], feats: [N,8])
            renders_cond/<scene_id>/view_YY/frame_XXXX.png  (512x512 RGBA)
            val/
                slat_latents/<scene_id>/frame_XXXX.npz
                renders_cond/<scene_id>/view_YY/frame_XXXX.png
        where view_00 is always the canonical front view, and view_01..view_{N-1}
        are Hammersley-sampled alternatives. ONE view is chosen per __getitem__
        call for the entire W-frame window.
    """
    def __init__(
        self,
        roots: str,
        *,
        window_size: int = 3,
        val_num_scenes: int = 5,
        val_num_frames: Optional[int] = None,
        image_size: int = 518,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        num_cond_views: int = 6,
        front_view_prob: float = 0.4,
        **kwargs,
    ):
        self.normalization = normalization
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
        super().__init__(
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.root = roots.split(',')[0]
        self.val_root = os.path.join(self.root, 'val')
        assert int(window_size) >= 1, f'window_size must be >= 1, got {window_size}'
        self.window_size = int(window_size)
        self.val_num_frames = val_num_frames
        self.image_size = image_size
        self.value_range = (0, 1)
        assert num_cond_views >= 1, f'num_cond_views must be >=1: {num_cond_views}'
        assert 0.0 <= front_view_prob <= 1.0, f'front_view_prob out of range: {front_view_prob}'
        # num_cond_views caps the max number of views used per scene.
        # If a scene has fewer views, all are used. If more, only the first
        # num_cond_views (sorted) are kept. If num_cond_views=1, only the
        # front view is used (no multi-view augmentation).
        self.num_cond_views = num_cond_views
        self.front_view_prob = front_view_prob

        # Per-scene available view folders: maps scene_id → sorted list of
        # actual view directory names (e.g., ['view_00', 'view_03', 'view_05']).
        # This handles missing intermediate views and per-scene view counts.
        self._scene_views: Dict[str, List[str]] = {}
        # Per-scene per-view set of frame indices PRESENT as PNG: used to
        # pick only views that actually contain every frame a sample needs.
        self._scene_view_frames: Dict[str, Dict[str, set]] = {}
        # Per-scene set of frame indices PRESENT as .npz (slat target).
        # A sample (scene, frame_idx) is usable only if its frame_idx is in
        # this set AND at least one cond view contains the curr+prev frames.
        self._scene_slat_frames: Dict[str, set] = {}

        def _scan_frames_for_scene(sid: str, slat_dir: str, cond_dir: str,
                                   num_cond_views: int):
            """Return (views, view_frame_sets, slat_frame_set) or None if scene
            has no slat files / no usable views. Also populates the
            per-scene caches as a side-effect."""
            slat_frames = set()
            for f in glob.glob(os.path.join(slat_dir, 'frame_*.npz')):
                try:
                    slat_frames.add(int(os.path.basename(f)[6:-4]))
                except ValueError:
                    continue
            if not slat_frames:
                return None
            views = sorted([
                d for d in os.listdir(cond_dir)
                if os.path.isdir(os.path.join(cond_dir, d)) and d.startswith('view_')
            ])[:num_cond_views]
            if not views:
                return None
            view_frame_sets: Dict[str, set] = {}
            for v in views:
                v_dir = os.path.join(cond_dir, v)
                frames_present = set()
                try:
                    names = os.listdir(v_dir)
                except OSError:
                    view_frame_sets[v] = frames_present
                    continue
                for fname in names:
                    if not (fname.startswith('frame_') and fname.endswith('.png')):
                        continue
                    try:
                        frames_present.add(int(fname[6:-4]))
                    except ValueError:
                        continue
                view_frame_sets[v] = frames_present
            # Drop views that have zero frames at all (corrupt directory).
            views = [v for v in views if view_frame_sets[v]]
            if not views:
                return None
            return views, view_frame_sets, slat_frames

        # ---- Train scenes from <root>/{encode_done, slat_latents, renders_cond} ----
        # Val samples were physically moved to <root>/val/, so their slat_latents
        # subdirs no longer exist under <root>/ and they are naturally excluded.
        encode_done_dir = os.path.join(self.root, 'encode_done')
        all_scene_ids = sorted([
            f.replace('.done', '') for f in os.listdir(encode_done_dir) if f.endswith('.done')
        ])
        train_scenes = []
        for sid in all_scene_ids:
            slat_dir = os.path.join(self.root, 'slat_latents', sid)
            cond_dir = os.path.join(self.root, 'renders_cond', sid)
            if not (os.path.isdir(slat_dir) and os.path.isdir(cond_dir)):
                continue
            scan = _scan_frames_for_scene(sid, slat_dir, cond_dir, num_cond_views)
            if scan is None:
                continue
            views, view_frame_sets, slat_frames = scan
            self._scene_views[sid] = views
            self._scene_view_frames[sid] = view_frame_sets
            self._scene_slat_frames[sid] = slat_frames
            train_scenes.append((sid, max(slat_frames) + 1))

        # ---- Val scenes from <root>/val/{slat_latents, renders_cond} ----
        # Cap to val_num_scenes scenes and val_num_frames frames per scene.
        # If either cap exceeds what's available, use everything available.
        val_scenes_all = []
        if os.path.isdir(self.val_root):
            val_slat = os.path.join(self.val_root, 'slat_latents')
            val_cond = os.path.join(self.val_root, 'renders_cond')
            if os.path.isdir(val_slat) and os.path.isdir(val_cond):
                for sid in sorted(os.listdir(val_slat)):
                    slat_dir = os.path.join(val_slat, sid)
                    cond_dir = os.path.join(val_cond, sid)
                    if not (os.path.isdir(slat_dir) and os.path.isdir(cond_dir)):
                        continue
                    scan = _scan_frames_for_scene(sid, slat_dir, cond_dir, num_cond_views)
                    if scan is None:
                        continue
                    views, view_frame_sets, slat_frames = scan
                    self._scene_views[sid] = views
                    self._scene_view_frames[sid] = view_frame_sets
                    self._scene_slat_frames[sid] = slat_frames
                    nf = max(slat_frames) + 1
                    if val_num_frames is not None:
                        nf = min(nf, val_num_frames)
                    val_scenes_all.append((sid, nf))
        if val_num_scenes > 0:
            self.val_scenes = val_scenes_all[:val_num_scenes]
        else:
            self.val_scenes = val_scenes_all

        # Build flat index for training: (scene_idx, start_idx).
        # Each entry indexes a W-frame window starting at `start_idx`,
        # covering consecutive frames {start_idx, start_idx + 1, ...,
        # start_idx + W - 1}. The dataset is a uniform distribution over all
        # valid W-frame windows across all scenes: scenes with N frames
        # contribute (N - W + 1) windows when N >= W, else zero. There is no
        # "anchor" concept; every frame in the window is treated equally by
        # the trainer and the model.
        self.scenes = []
        self.flat_index = []
        n_skipped = 0
        for sid, num_frames in train_scenes:
            slat_frames = self._scene_slat_frames[sid]
            view_frame_sets = self._scene_view_frames[sid]
            scene_idx = len(self.scenes)
            self.scenes.append((sid, num_frames))
            # A scene shorter than W frames contributes no windows.
            n_windows = num_frames - self.window_size + 1
            if n_windows <= 0:
                n_skipped += num_frames
                continue
            for start_idx in range(n_windows):
                required = {start_idx + k for k in range(self.window_size)}
                # Every frame in the window must have its SLat .npz on disk.
                if not required.issubset(slat_frames):
                    n_skipped += 1
                    continue
                # At least one cond view must contain every frame in the window.
                if not any(required.issubset(view_frame_sets[v]) for v in self._scene_views[sid]):
                    n_skipped += 1
                    continue
                self.flat_index.append((scene_idx, start_idx))

        # For load balancing — estimate token count per sample.
        # Use the smallest available frame index instead of hardcoding 0000,
        # since some scenes may be missing frame_0000.npz.
        self._token_estimates = {}
        for scene_idx, (sid, nf) in enumerate(self.scenes):
            slat_frames_sid = self._scene_slat_frames[sid]
            if not slat_frames_sid:
                self._token_estimates[scene_idx] = 0
                continue
            first_idx = min(slat_frames_sid)
            first_frame = os.path.join(self.root, 'slat_latents', sid,
                                       f'frame_{first_idx:04d}.npz')
            try:
                data = np.load(first_frame)
                self._token_estimates[scene_idx] = data['coords'].shape[0]
            except (FileNotFoundError, OSError, KeyError) as e:
                print(f"  [warn] token estimate failed for {sid} ({first_frame}): {e!r}")
                self._token_estimates[scene_idx] = 0

        self.loads = [
            self._token_estimates[si] for si, _ in self.flat_index
        ]

        view_counts = [len(self._scene_views.get(sid, [])) for sid, _ in train_scenes]
        view_min = min(view_counts) if view_counts else 0
        view_max = max(view_counts) if view_counts else 0
        print(f"TemporalSLat: "
              f"{len(self.scenes)} train / {len(self.val_scenes)} val scenes "
              f"(val cap: val_num_scenes={val_num_scenes}, val_num_frames={val_num_frames}), "
              f"{len(self.flat_index)} train frames "
              f"({n_skipped} skipped: scene shorter than W, "
              f"missing slat, or no view covers the window), "
              f"window_size={self.window_size}, "
              f"views_per_scene={view_min}-{view_max}, front_view_prob={self.front_view_prob}")
        for sid, nf in self.val_scenes:
            print(f"  val: {sid} ({nf} frames)")

    def __len__(self):
        return len(self.flat_index)

    def _pick_view(self, scene_id: str,
                   required_frames: Optional[set] = None) -> Optional[str]:
        """Pick a conditioning view folder name that contains every frame in
        `required_frames`. Returns None if no view satisfies the requirement.

        Sampling preserves the front-view-bias: the front view (sorted first)
        is chosen with `front_view_prob` whenever it is among the valid views.
        Otherwise samples uniformly among the non-front valid views.
        """
        views = self._scene_views.get(scene_id, ['view_00'])
        view_frame_sets = self._scene_view_frames.get(scene_id, {})
        if required_frames is None:
            valid = list(views)
        else:
            valid = [v for v in views if required_frames.issubset(view_frame_sets.get(v, set()))]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        front = views[0]
        front_valid = front in valid
        if front_valid and np.random.random() < self.front_view_prob:
            return front
        non_front_valid = [v for v in valid if v != front]
        if not non_front_valid:
            return front
        return non_front_valid[np.random.randint(len(non_front_valid))]

    def _load_frame(self, scene_id: str, frame_idx: int,
                    view_name: str = 'view_00', from_val: bool = False):
        """Load one frame's latent coords/feats and conditioning image.

        view_name is the actual view folder name (e.g., 'view_00', 'view_03'),
        NOT an integer index. This handles non-contiguous view numbering.
        from_val=True reads from <root>/val/, otherwise from <root>/.
        """
        base = self.val_root if from_val else self.root
        slat_path = os.path.join(base, 'slat_latents', scene_id, f'frame_{frame_idx:04d}.npz')
        data = np.load(slat_path)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std

        img_path = os.path.join(
            base, 'renders_cond', scene_id,
            view_name, f'frame_{frame_idx:04d}.png',
        )
        image = Image.open(img_path)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)

        return coords, feats, image

    def __getitem__(self, index):
        # Retry up to a small number of times on missing files / corrupt PNGs
        # that slip past the init-time pre-filter (race conditions, mid-flight
        # rsync, etc.). Each retry resamples a new random index.
        MAX_RETRIES = 8
        last_err: Optional[Exception] = None
        for retry in range(MAX_RETRIES):
            try:
                scene_idx, start_idx = self.flat_index[index]
                scene_id, num_frames = self.scenes[scene_idx]

                # W consecutive frames: [start_idx, ..., start_idx + W - 1].
                required = {start_idx + k for k in range(self.window_size)}

                view_name = self._pick_view(scene_id, required_frames=required)
                if view_name is None:
                    raise FileNotFoundError(
                        f"No view in scene {scene_id} contains all required frames {required}"
                    )

                # Load every frame in the window. Ordered oldest -> newest by
                # slot position; the model receives `window_frames[k]` as the
                # frame at window-local position `k`.
                window_coords: List[torch.Tensor] = []
                window_feats: List[torch.Tensor] = []
                window_token_counts: List[int] = []
                window_conds: List[torch.Tensor] = []
                for k in range(self.window_size):
                    frame_idx = start_idx + k
                    c, f, cnd = self._load_frame(scene_id, frame_idx, view_name=view_name)
                    window_coords.append(c)
                    window_feats.append(f)
                    window_token_counts.append(c.shape[0])
                    window_conds.append(cnd)

                return {
                    'window_coords': window_coords,
                    'window_feats': window_feats,
                    'window_token_counts': window_token_counts,
                    'window_conds': window_conds,
                    'window_size': self.window_size,
                }
            except (FileNotFoundError, OSError, ValueError) as e:
                last_err = e
                # Resample a different index for the next attempt.
                index = int(np.random.randint(len(self.flat_index)))
        raise RuntimeError(
            f"TemporalSLat: failed to load any sample "
            f"after {MAX_RETRIES} retries. Last error: {last_err!r}"
        )

    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate W-frame windows from B scenes into a flat (B*W) batch.

        Every sample contributes exactly W frames. All B*W frames are packed
        flat into a single SparseTensor (`frames_x0`) with batch dim B*W;
        `frames_to_scene` and `frames_position` index back to (scene, window
        slot). The trainer groups frames by scene to assemble each scene's
        W-frame list in slot order.

        Returns dict with:
            frames_x0: SparseTensor with batch dim B*W (each (scene, slot)
                       is one batch element).
            frames_to_scene: (B*W,) long, scene index in [0, B) for each frame.
            frames_position: (B*W,) long, window-local slot (0..W-1) for each
                             frame.
            frames_cond_images: (B*W, 3, H, W) cond images matched to
                                `frames_x0` batch order.
            num_scenes: B.
            window_size: W.
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            # Load = sum of token counts across all W frames in the window.
            group_idx = load_balanced_group_indices(
                [sum(b['window_token_counts']) for b in batch], split_size
            )

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            B = len(sub_batch)
            W = sub_batch[0]['window_size']

            flat_coords = []
            flat_feats = []
            flat_layout = []
            flat_to_scene = []
            flat_positions = []
            flat_cond_images = []
            flat_idx = 0
            start = 0
            for s_i, b in enumerate(sub_batch):
                assert b['window_size'] == W, (
                    f"window_size mismatch in batch: {b['window_size']} vs {W}"
                )
                for slot in range(W):
                    c = b['window_coords'][slot]
                    f = b['window_feats'][slot]
                    flat_coords.append(torch.cat([
                        torch.full((c.shape[0], 1), flat_idx, dtype=torch.int32),
                        c,
                    ], dim=-1))
                    flat_feats.append(f)
                    flat_layout.append(slice(start, start + c.shape[0]))
                    flat_to_scene.append(s_i)
                    flat_positions.append(slot)
                    flat_cond_images.append(b['window_conds'][slot])
                    start += c.shape[0]
                    flat_idx += 1

            frames_x0 = SparseTensor(
                coords=torch.cat(flat_coords),
                feats=torch.cat(flat_feats),
            )
            frames_x0._shape = torch.Size([flat_idx, *sub_batch[0]['window_feats'][0].shape[1:]])
            frames_x0.register_spatial_cache('layout', flat_layout)

            pack = {
                'frames_x0': frames_x0,
                'frames_to_scene': torch.tensor(flat_to_scene, dtype=torch.long),
                'frames_position': torch.tensor(flat_positions, dtype=torch.long),
                'frames_cond_images': torch.stack(flat_cond_images),
                'num_scenes': B,
                'window_size': W,
            }
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    PIN_MEMORY = False

    @torch.no_grad()
    def visualize_sample(self, x_0):
        """Override SLatVisMixin.visualize_sample with FIXED cameras so that
        sample_gt and sample renders use the same viewpoints."""
        import utils3d.torch
        from ..utils.render_utils import get_renderer
        x_0 = x_0 if hasattr(x_0, 'feats') else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())

        # Fixed cameras: 4 yaws around the object, fixed pitch
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pitch = [np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6]

        exts, ints = [], []
        for yaw, p in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(p),
                np.cos(yaw) * np.cos(p),
                np.sin(p),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig, torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for rep in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1),
                      512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        return torch.stack(images)

    def _load_val_scene(self, sid: str, from_val: bool = True):
        """Load all frames of a single validation scene from <root>/val/ (or
        <root>/ when from_val=False, used as the no-val-scenes fallback).
        Picks the first view that contains every requested frame for a
        reproducible snapshot; capped to self.val_num_frames frames if set."""
        base = self.val_root if from_val else self.root
        slat_dir = os.path.join(base, 'slat_latents', sid)
        num_frames = len(glob.glob(os.path.join(slat_dir, 'frame_*.npz')))
        if self.val_num_frames is not None:
            num_frames = min(num_frames, self.val_num_frames)

        views = self._scene_views.get(sid, ['view_00'])
        view_frame_sets = self._scene_view_frames.get(sid, {})
        slat_frames = self._scene_slat_frames.get(sid, set(range(num_frames)))

        # Trim num_frames to the largest contiguous prefix [0..n) such that
        # frame i's slat AND at least one view's PNG is present for every i.
        valid_n = num_frames
        for i in range(num_frames):
            if i not in slat_frames:
                valid_n = i
                break
            if not any(i in view_frame_sets.get(v, set()) for v in views):
                valid_n = i
                break
        if valid_n != num_frames:
            print(f"  [warn] val scene {sid}: trimming {num_frames} -> {valid_n} frames "
                  f"(missing slat or cond PNGs from frame {valid_n})")
            num_frames = valid_n

        # Pick the first view that has ALL valid_n frames; fall back to the
        # front view if none cover everything (shouldn't happen given the
        # contiguous-prefix trim above, but kept defensively).
        required = set(range(num_frames))
        valid_view = None
        for v in views:
            if required.issubset(view_frame_sets.get(v, set())):
                valid_view = v
                break
        if valid_view is None:
            valid_view = views[0]

        all_coords = []
        all_feats = []
        all_cond = []
        for i in range(num_frames):
            coords, feats, cond = self._load_frame(sid, i, view_name=valid_view, from_val=from_val)
            all_coords.append(coords)
            all_feats.append(feats)
            all_cond.append(cond)

        return {
            'scene_id': sid,
            'all_coords': all_coords,
            'all_feats': all_feats,
            'all_cond': torch.stack(all_cond),
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
        return (
            f"TemporalSLat\n"
            f"  - Train scenes: {len(self.scenes)}\n"
            f"  - Val scenes: {len(self.val_scenes)}\n"
            f"  - Total train frames: {len(self.flat_index)}\n"
            f"  - Window size: {self.window_size}\n"
            f"  - Image size: {self.image_size}\n"
            f"  - Normalization: {self.normalization is not None}\n"
            f"  - Cond views: {self.num_cond_views} (front_view_prob={self.front_view_prob})"
        )
