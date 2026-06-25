"""
Diffusion Forcing trainer for temporal SLat flow matching.

Symmetric W-frame denoising: all W frames in a window are noised at
independent levels and denoised jointly. There is no "current" vs "previous"
distinction; every frame is one window slot, every frame is noisy, and the
model predicts velocity for all tokens simultaneously.

CFG drop is per-scene at training: when a scene fires (Bernoulli with
prob ``p_uncond``), ALL W frames in that scene's window get their cond
features zeroed together. This matches the dual-KV inference path, where
the negative-CFG branch attends to a KV cache that was built with
``neg_cond=zeros`` across the full window.

Inference (snapshot) uses ``FlowEulerKVCacheGuidanceIntervalSampler`` —
shared with the SS diffforcing trainer — with ``rescale_t`` + ``cfg_interval``
support. CFG formula via the sampler is Form B,
``(1+cfg)·v_pos − cfg·v_neg``, algebraically identical to Form A.

``snapshot_modes`` controls which validation rollouts run per scene; the
default ``["ar_kvcache"]`` runs the AR rollout with KV cache. Supported
modes: ``ar_kvcache``, ``tf_kvcache``, ``ar``, ``tf``.
"""
from typing import *
import os
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...modules.sparse.basic import SparseTensor
from ...utils.data_utils import cycle, BalancedResumableSampler
from ...utils.video_to_4d_utils import ShiftingBank
from .diffusion_forcing_base import DiffusionForcingTrainerBase
from .mixins.image_conditioned import ImageConditionedMixin


class DiffusionForcingSLatTrainer(DiffusionForcingTrainerBase):
    """
    SLat diffusion-forcing trainer: B scenes × W frames per step.

    SLat-specific args (the rest live in ``DiffusionForcingTrainerBase``):
        coord_drop_prob: per-token sparse-coord dropout rate. Models the
            inference scenario where some frames have SS prediction error.
        coord_drop_num: how many slots in each W-window receive the
            token-level drop. ``None`` (default) = drop on every slot
            (symmetric augmentation). ``0`` = no drop. ``1..W`` = that many
            random slots per scene drop; the rest are kept clean.
    """

    DDP_STATIC_GRAPH = False
    _supported_modes = ("ar_kvcache", "tf_kvcache", "ar", "tf")

    def __init__(
        self,
        *args,
        coord_drop_prob: float = 0.0,
        coord_drop_num: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coord_drop_prob = float(coord_drop_prob)
        if coord_drop_num is None:
            self.coord_drop_num = self.window_size      # drop all
        else:
            self.coord_drop_num = int(coord_drop_num)
        assert 0 <= self.coord_drop_num <= self.window_size, (
            f'coord_drop_num must be in [0, window_size={self.window_size}], '
            f'got {self.coord_drop_num}'
        )
        self._record_val_scenes()

    def _record_val_scenes(self):
        if not getattr(self, 'is_master', True):
            return
        val_scenes = getattr(self.dataset, 'val_scenes', None)
        if val_scenes is None:
            return
        import json
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'val_scenes.json'), 'w') as f:
            json.dump({
                'val_num_scenes': len(val_scenes),
                'num_train_scenes': len(getattr(self.dataset, 'scenes', [])),
                'val_scenes': [{'scene_id': s, 'num_frames': n} for s, n in val_scenes],
            }, f, indent=2)

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu,
        )
        pin_memory = getattr(self.dataset, 'PIN_MEMORY', True)
        env_workers = os.environ.get('DATALOADER_NUM_WORKERS')
        num_workers = int(env_workers) if env_workers else min(
            8, max(1, int(np.ceil(os.cpu_count() / max(1, torch.cuda.device_count()))))
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _noise_sparse(self, x_0: SparseTensor, t: torch.Tensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
        """Apply flow-matching noise to a SparseTensor at per-batch-element t.
        Returns (x_t, noise, target_velocity)."""
        noise = x_0.replace(torch.randn_like(x_0.feats))
        from ...modules.sparse.basic import sparse_batch_broadcast
        t_expand = sparse_batch_broadcast(x_0, t.unsqueeze(-1))
        x_t_feats = (1 - t_expand) * x_0.feats + (self.sigma_min + (1 - self.sigma_min) * t_expand) * noise.feats
        x_t = x_0.replace(x_t_feats)
        target = x_0.replace((1 - self.sigma_min) * noise.feats - x_0.feats)
        return x_t, noise, target

    def training_losses(
        self,
        frames_x0: SparseTensor = None,
        frames_to_scene: torch.Tensor = None,
        frames_position: torch.Tensor = None,
        frames_cond_images: torch.Tensor = None,
        num_scenes: int = None,
        window_size: int = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """Symmetric W-frame diffusion forcing (see module docstring)."""
        B = int(num_scenes)
        W = int(window_size)
        device = frames_x0.device
        n_flat = frames_x0.shape[0]
        assert n_flat == B * W, (
            f"frames_x0 should have B*W={B*W} batch elements; got {n_flat}"
        )

        # Encode all B*W cond images at once: (B*W, L, D).
        frame_cond_encoded = self.encode_image(frames_cond_images.to(device))

        # Per-scene CFG drop: fire one Bernoulli per scene; on fire, zero ALL
        # W cond features for that scene. Torch RNG (DDP-safe, device-local).
        if self.p_uncond > 0:
            drop_mask = torch.rand(B, device=device) < self.p_uncond     # (B,)
            per_frame_drop = drop_mask[frames_to_scene]                  # (B*W,)
            if per_frame_drop.any():
                keep = (~per_frame_drop).to(frame_cond_encoded.dtype)
                keep = keep.view(-1, *([1] * (frame_cond_encoded.dim() - 1)))
                frame_cond_encoded = frame_cond_encoded * keep

        # Per-(scene, slot) noise level — base helper respects independent_t.
        t_all = self._sample_t_per_frame(B, W, device)  # (B, W)

        all_pred_feats = []
        all_target_feats = []

        # `collate_fn` packs frames in (scene-major, slot-minor) order, so
        # scene b's W frames are at flat indices [b*W, b*W+1, ..., b*W+W-1]
        # in slot order. (Contract enforced by the dataset.)
        for b in range(B):
            scene_indices = range(b * W, (b + 1) * W)

            # Pick which slots RECEIVE coord_drop for this scene.
            if self.coord_drop_num >= W:
                drop_slots = set(range(W))
            elif self.coord_drop_num <= 0:
                drop_slots = set()
            else:
                drop_slots = set(
                    np.random.choice(W, size=self.coord_drop_num, replace=False).tolist()
                )

            window_frames = []
            window_conds = []
            for slot, k in enumerate(scene_indices):
                sl = frames_x0.layout[k]
                c = frames_x0.coords[sl.start:sl.stop].clone()
                c[:, 0] = 0
                f = frames_x0.feats[sl.start:sl.stop]
                if self.coord_drop_prob > 0 and slot in drop_slots:
                    keep = torch.rand(c.shape[0], device=device) > self.coord_drop_prob
                    if not keep.any():
                        keep[0] = True
                    c = c[keep]
                    f = f[keep]
                window_frames.append(SparseTensor(coords=c, feats=f))
                window_conds.append(frame_cond_encoded[k:k+1])

            t_per_frame = t_all[b]  # (W,)
            noised_frames = []
            target_frames = []
            for i in range(W):
                t_i = t_per_frame[i].unsqueeze(0)
                x_t_i, _, target_i = self._noise_sparse(window_frames[i], t_i)
                noised_frames.append(x_t_i)
                target_frames.append(target_i)

            frame_indices = list(range(W))
            pred = self.training_models['denoiser'](
                noised_frames, t_per_frame, window_conds[-1],
                frame_indices=frame_indices,
                per_frame_cond=window_conds,
            )
            all_pred_feats.append(pred.feats)
            all_target_feats.append(torch.cat([t.feats for t in target_frames]))

        terms = edict()
        terms["mse"] = F.mse_loss(
            torch.cat(all_pred_feats), torch.cat(all_target_feats),
        )
        terms["loss"] = terms["mse"]
        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot inference: dual cond/uncond KV-cache rollout via the shared
    # FlowEulerKVCacheGuidanceIntervalSampler.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference_pass(self, gt_results, cond_images, num_frames,
                            kv_source, verbose=False):
        """Dual KV-cache CFG rollout. ``kv_source='self'`` -> ar_kvcache;
        ``kv_source='gt'`` -> tf_kvcache."""
        model = self.models['denoiser']
        W = model.window_size
        sampler = self.get_sampler()
        sample_results: List[SparseTensor] = []
        pos_bank = ShiftingBank(max_window=max(0, W - 1))
        neg_bank = ShiftingBank(max_window=max(0, W - 1))
        all_cond_encoded = [self.encode_image(img) for img in cond_images]

        for frame_idx in range(num_frames):
            x_0 = gt_results[frame_idx]
            cond = all_cond_encoded[frame_idx]
            neg_cond = torch.zeros_like(cond)
            pos_window = pos_bank.window()
            neg_window = neg_bank.window()

            noise = x_0.replace(torch.randn_like(x_0.feats))
            res = sampler.sample(
                model, noise=noise, cond=cond, neg_cond=neg_cond,
                frame_idx=frame_idx,
                pos_kv_caches=pos_window, neg_kv_caches=neg_window,
                steps=self.snapshot_steps, rescale_t=self.rescale_t,
                cfg_strength=self.cfg_strength, cfg_interval=self.cfg_interval,
                verbose=verbose,
            )
            sample_results.append(res.samples)

            src = gt_results if kv_source == 'gt' else sample_results
            denoised = src[frame_idx]
            pos_bank.append(model.build_kv_cache(
                denoised, t_noise=0.0, frame_idx=frame_idx, cond=cond,
                sigma_min=self.sigma_min,
                prev_kv_caches=pos_window or None,
            ))
            neg_bank.append(model.build_kv_cache(
                denoised, t_noise=0.0, frame_idx=frame_idx, cond=neg_cond,
                sigma_min=self.sigma_min,
                prev_kv_caches=neg_window or None,
            ))

        return sample_results

    def _build_gt_tensors(self, all_coords, all_feats, all_cond, num_frames):
        gt_results = []
        cond_images = []
        for frame_idx in range(num_frames):
            coords_f = all_coords[frame_idx].cuda()
            feats_f = all_feats[frame_idx].cuda()
            cond_img = all_cond[frame_idx].unsqueeze(0).cuda()
            batch_idx = torch.zeros(coords_f.shape[0], 1, dtype=torch.int32, device=coords_f.device)
            full_coords = torch.cat([batch_idx, coords_f], dim=-1)
            gt_results.append(SparseTensor(coords=full_coords, feats=feats_f))
            cond_images.append(cond_img)
        return gt_results, cond_images

    @torch.no_grad()
    def _run_inference_pass_no_kvcache(self, gt_results, cond_images,
                                       num_frames, kv_source, verbose=False):
        """No-KV-cache rollout via the training forward path."""
        model = self.models['denoiser']
        W = model.window_size
        sample_results: List[SparseTensor] = []
        all_cond_encoded = [self.encode_image(img) for img in cond_images]

        steps = self.snapshot_steps
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = self.rescale_t * t_seq / (1 + (self.rescale_t - 1) * t_seq)

        for frame_idx in range(num_frames):
            x_0 = gt_results[frame_idx]
            device = x_0.device
            cond = all_cond_encoded[frame_idx]
            neg_cond = torch.zeros_like(cond)

            window_start = max(0, frame_idx - (W - 1))
            wnd_abs_indices = list(range(window_start, frame_idx))
            src = gt_results if kv_source == 'gt' else sample_results

            wnd_frames_clean = []
            wnd_conds = []
            for j in wnd_abs_indices:
                wnd_st = src[j]
                wc = wnd_st.coords.clone()
                wc[:, 0] = 0
                wnd_frames_clean.append(SparseTensor(coords=wc.clone(), feats=wnd_st.feats))
                wnd_conds.append(all_cond_encoded[j])

            n_wnd = len(wnd_abs_indices)
            frame_indices = list(range(n_wnd + 1))
            x_t = x_0.replace(torch.randn_like(x_0.feats))

            for step in range(steps):
                t_val = float(t_seq[step])
                dt = float(t_seq[step] - t_seq[step + 1])
                t_anchor = torch.tensor(t_val, device=device)

                anchor_coords = x_t.coords.clone()
                anchor_coords[:, 0] = 0
                anchor_frame = SparseTensor(coords=anchor_coords, feats=x_t.feats)
                n_anchor = anchor_frame.feats.shape[0]

                x_frames_clean = wnd_frames_clean + [anchor_frame]
                if n_wnd > 0:
                    t_per_clean = torch.cat([
                        torch.zeros(n_wnd, device=device),
                        t_anchor.unsqueeze(0),
                    ])
                else:
                    t_per_clean = t_anchor.unsqueeze(0)
                per_frame_cond_pos = wnd_conds + [cond]
                per_frame_cond_neg = [neg_cond] * (n_wnd + 1)

                pred_cond = model(
                    x_frames_clean, t_per_clean, cond,
                    frame_indices=frame_indices,
                    per_frame_cond=per_frame_cond_pos,
                )
                v_cond = pred_cond.feats[-n_anchor:]
                if self.cfg_interval[0] <= t_val <= self.cfg_interval[1]:
                    pred_uncond = model(
                        x_frames_clean, t_per_clean, neg_cond,
                        frame_indices=frame_indices,
                        per_frame_cond=per_frame_cond_neg,
                    )
                    v_uncond = pred_uncond.feats[-n_anchor:]
                    v = v_cond + self.cfg_strength * (v_cond - v_uncond)
                else:
                    v = v_cond
                x_t = x_t.replace(x_t.feats - dt * v)

            sample_results.append(x_t)

        return sample_results

    # ------------------------------------------------------------------
    # Perceptual metric helpers (LPIPS / CLIP / DreamSim on rendered samples)
    # ------------------------------------------------------------------
    def _get_metric_models(self):
        """Lazy-load LPIPS / CLIP / DreamSim once and cache on self."""
        if getattr(self, '_metric_models', None) is not None:
            return self._metric_models
        import lpips
        import open_clip
        from dreamsim import dreamsim
        device = torch.device('cuda')
        lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        for p in lpips_model.parameters():
            p.requires_grad = False
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai',
        )
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        dreamsim_model, dreamsim_preprocess = dreamsim(
            pretrained=True, device=str(device),
        )
        dreamsim_model.eval()
        self._metric_models = {
            'device': device,
            'lpips': lpips_model,
            'clip': clip_model,
            'clip_preprocess': clip_preprocess,
            'dreamsim_model': dreamsim_model,
            'dreamsim_preprocess': dreamsim_preprocess,
        }
        return self._metric_models

    @torch.no_grad()
    def _render_4view_tensor_pil(self, samples: List[SparseTensor]):
        """Render a list of per-frame SparseTensors as 4-view tiles."""
        sample_st = sp.sparse_cat(samples)
        tiled = self.dataset.visualize_sample(sample_st)   # [T, 3, 1024, 1024]
        T = tiled.shape[0]
        h, w = 512, 512
        views = torch.stack([
            tiled[:, :, 0:h,   0:w  ],
            tiled[:, :, 0:h,   w:2*w],
            tiled[:, :, h:2*h, 0:w  ],
            tiled[:, :, h:2*h, w:2*w],
        ], dim=1)
        V = 4
        flat = views.reshape(T * V, 3, h, w).clamp(0, 1).contiguous()
        arr = (flat.float().cpu() * 255).to(torch.uint8).movedim(1, -1).numpy()
        pil_list = [Image.fromarray(x) for x in arr]
        return flat, pil_list, T, V

    @torch.no_grad()
    def _perceptual_batch(self, pred_tensor, pred_pil, gt_tensor, gt_pil,
                          batch_size: int = 24):
        """Mean LPIPS / CLIP / DreamSim over the (T·V) rendered crops."""
        m = self._get_metric_models()
        device = m['device']
        N = pred_tensor.shape[0]
        lp_chunks, cl_chunks, ds_chunks = [], [], []
        for i in range(0, N, batch_size):
            sl = slice(i, min(i + batch_size, N))
            a = (pred_tensor[sl].to(device) * 2 - 1).float()
            b = (gt_tensor[sl].to(device) * 2 - 1).float()
            lp_chunks.append(m['lpips'](a, b).view(-1).cpu())
            a_clip = torch.stack([m['clip_preprocess'](x) for x in pred_pil[sl]]).to(device)
            b_clip = torch.stack([m['clip_preprocess'](x) for x in gt_pil[sl]]).to(device)
            fa = F.normalize(m['clip'].encode_image(a_clip).float(), dim=-1)
            fb = F.normalize(m['clip'].encode_image(b_clip).float(), dim=-1)
            cl_chunks.append((fa * fb).sum(-1).cpu())
            a_ds = torch.cat([m['dreamsim_preprocess'](x) for x in pred_pil[sl]]).to(device)
            b_ds = torch.cat([m['dreamsim_preprocess'](x) for x in gt_pil[sl]]).to(device)
            ds_chunks.append(m['dreamsim_model'](a_ds, b_ds).view(-1).cpu())
        return (
            torch.cat(lp_chunks).mean().item(),
            torch.cat(cl_chunks).mean().item(),
            torch.cat(ds_chunks).mean().item(),
        )

    # ------------------------------------------------------------------
    # Subclass hook for the shared run_snapshot
    # ------------------------------------------------------------------
    def _dispatch_inference(self, mode, scene, n, verbose=False):
        """Return per-metric value lists for one (mode, scene) pair.

        Latent MSE is one value per frame; perceptual metrics are one
        scene-level mean appended once per scene.
        """
        gt_list, cond_list = self._build_gt_tensors(
            scene['all_coords'], scene['all_feats'], scene['all_cond'], n,
        )
        kv_source = 'gt' if mode.startswith('tf') else 'self'
        if mode.endswith('_kvcache'):
            samples = self._run_inference_pass(
                gt_list, cond_list, n, kv_source, verbose=verbose,
            )
        else:
            samples = self._run_inference_pass_no_kvcache(
                gt_list, cond_list, n, kv_source, verbose=verbose,
            )

        out: Dict[str, List[float]] = {
            'mse': [F.mse_loss(samples[i].feats, gt_list[i].feats).item()
                    for i in range(n)],
        }
        # Perceptual on rendered 4-view tiles. GT render reused across modes
        # via an on-self cache keyed by the gt_list identity.
        cache = getattr(self, '_gt_render_cache', None)
        if cache is None or cache[0] is not gt_list:
            gt_tensor, gt_pil, T, V = self._render_4view_tensor_pil(gt_list)
            self._gt_render_cache = (gt_list, gt_tensor, gt_pil, T, V)
        else:
            _, gt_tensor, gt_pil, T, V = cache
        pred_tensor, pred_pil, T_p, V_p = self._render_4view_tensor_pil(samples)
        assert T_p == T and V_p == V
        lp, cl, ds = self._perceptual_batch(
            pred_tensor, pred_pil, gt_tensor, gt_pil,
        )
        out['lpips'] = [lp]
        out['clip'] = [cl]
        out['dreamsim'] = [ds]
        return out


class DiffusionForcingSLatImageConditionedTrainer(
    ImageConditionedMixin, DiffusionForcingSLatTrainer
):
    """Diffusion forcing trainer with DINOv2 image conditioning."""
    pass
