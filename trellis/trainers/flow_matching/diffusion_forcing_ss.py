"""Trainer for stage-1 diffusion-forcing causal temporal flow matching (SS).

Processes ``window_size`` consecutive frames per training step with block-causal
self-attention; each frame gets an independent noise level. Validation /
snapshot runs with a sliding **dual** KV cache (positive cache built with cond,
negative cache built with neg_cond) via
``FlowEulerKVCacheGuidanceIntervalSampler`` — the same full-window-CFG
convention the SLat stage and the inference pipeline use.

``snapshot_modes`` controls which validation rollouts run per scene. Default
``["ar_kvcache"]`` runs AR with the KV cache. Supported: ``ar_kvcache`` (caches
from generated samples) and ``tf_kvcache`` (caches from GT). The no-cache modes
(``ar``, ``tf``) would require a separate dense multi-frame forward path; they
raise ``NotImplementedError`` for SS for now.
"""
from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

from ...utils.video_to_4d_utils import ShiftingBank
from .diffusion_forcing_base import DiffusionForcingTrainerBase
from .mixins.image_conditioned import ImageConditionedMixin


class DiffusionForcingSSTrainer(DiffusionForcingTrainerBase):
    """
    SS diffusion-forcing trainer.

    All snapshot dials (``window_size``, ``snapshot_steps``, ``snapshot_modes``,
    ``rescale_t``, ``cfg_interval``, etc.) live in
    :class:`DiffusionForcingTrainerBase`. The number of frames rolled out per
    val scene at snapshot is read from ``dataset.val_num_frames`` (shared with
    SLat); when null, the full scene is rolled out.
    """

    # SS doesn't implement a dense no-cache multi-frame forward path; only the
    # KV-cache variants are wired up.
    _supported_modes = ("ar_kvcache", "tf_kvcache")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        num_frames=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """Single-step causal training over a ``window_size``-frame chunk.

        Args:
            x_0: ``(B, T, 8, 16, 16, 16)`` full scene GT latents.
            cond: ``(B, T, 3, 518, 518)`` full scene cond images.
        """
        B, T = x_0.shape[:2]
        W = self.window_size
        device = x_0.device
        assert T >= W, (
            f"Scene has only {T} frames but window_size={W}. "
            "Dataset should filter scenes with <window_size frames."
        )

        # Per-sequence start index: each batch element samples its own window
        # start independently so the gradient sees B distinct (scene, start)
        # pairs per step.
        if T == W:
            s = torch.zeros(B, dtype=torch.long, device=device)
        else:
            s = torch.randint(0, T - W + 1, (B,), device=device)
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, W)
        frame_idx_full = s.view(B, 1) + torch.arange(W, device=device).view(1, W)
        x_0_win = x_0[batch_idx, frame_idx_full]    # (B, W, 8, 16, 16, 16)
        cond_win = cond[batch_idx, frame_idx_full]  # (B, W, 3, 518, 518)

        # Per-(scene, slot) noise levels via the shared base helper.
        t = self._sample_t_per_frame(B, W, device)  # (B, W)

        noise = torch.randn_like(x_0_win)
        t_broadcast = t.view(B, W, 1, 1, 1, 1)
        x_t = (1 - t_broadcast) * x_0_win + (
            self.sigma_min + (1 - self.sigma_min) * t_broadcast
        ) * noise

        # Encode cond per-frame via DINOv2 (bypassing CFG dropout in the
        # mixin so we can apply per-window full-drop below).
        cond_flat = cond_win.reshape(B * W, *cond_win.shape[2:])  # (B*W, 3, H, W)
        cond_encoded = self.encode_image(cond_flat)               # (B*W, L_c, D_c)
        L_c, D_c = cond_encoded.shape[1], cond_encoded.shape[2]
        cond_encoded = cond_encoded.view(B, W, L_c, D_c)

        # Per-scene full-window CFG drop (matches dual-KV uncond inference).
        if self.p_uncond > 0:
            drop_mask = (torch.rand(B, device=device) < self.p_uncond)  # (B,)
            cond_encoded = torch.where(
                drop_mask.view(B, 1, 1, 1),
                torch.zeros_like(cond_encoded),
                cond_encoded,
            )

        frame_indices = torch.arange(W, device=device, dtype=torch.long)
        frame_indices = frame_indices.unsqueeze(0).expand(B, W)

        pred = self.training_models['denoiser'](
            x_t, t, cond_encoded, frame_indices=frame_indices,
        )
        assert pred.shape == x_0_win.shape, f"pred {pred.shape} vs gt {x_0_win.shape}"

        target = (1 - self.sigma_min) * noise - x_0_win
        loss = F.mse_loss(pred, target)

        with torch.no_grad():
            per_frame_mse = [
                F.mse_loss(pred[:, i], target[:, i]).item() for i in range(W)
            ]

        terms = edict()
        terms["mse"] = loss
        terms["loss"] = loss
        terms["start_frame"] = float(s.float().mean().item())
        for i, v in enumerate(per_frame_mse):
            terms[f"frame_{i}_mse"] = float(v)
        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot inference (dual cond/uncond KV cache via the shared sampler)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference_pass(self, data: Dict, num_frames: int,
                            kv_source: str = 'self',
                            verbose: bool = False) -> Dict:
        """Dual cond/uncond KV-cache rollout via
        ``FlowEulerKVCacheGuidanceIntervalSampler``.

        ``kv_source='self'`` -> AR; ``kv_source='gt'`` -> TF. Mirrors
        ``DiffusionForcingSLatTrainer._run_inference_pass``
        on the dense SS latent path. ``num_frames`` is the per-scene cap
        computed by ``DiffusionForcingTrainerBase._frame_count`` (which
        reads ``dataset.val_num_frames``).
        """
        model = self.models['denoiser']
        base = model.base_model
        in_ch = base.in_channels
        reso = base.resolution
        W = self.window_size
        sampler = self.get_sampler()

        cond_features = [
            self.encode_image(data['cond'][i].unsqueeze(0).cuda())
            for i in range(num_frames)
        ]
        sample_gt = [data['x_0'][i].unsqueeze(0).cuda() for i in range(num_frames)]

        pos_bank = ShiftingBank(max_window=max(0, W - 1))
        neg_bank = ShiftingBank(max_window=max(0, W - 1))
        sample_pred: List[torch.Tensor] = []

        for frame_idx in range(num_frames):
            cond = cond_features[frame_idx]
            neg_cond = torch.zeros_like(cond)
            pos_window = pos_bank.window()
            neg_window = neg_bank.window()

            noise = torch.randn(1, in_ch, reso, reso, reso, device=cond.device)
            res = sampler.sample(
                model, noise=noise, cond=cond, neg_cond=neg_cond,
                frame_idx=frame_idx,
                pos_kv_caches=pos_window, neg_kv_caches=neg_window,
                steps=self.snapshot_steps, rescale_t=self.rescale_t,
                cfg_strength=self.cfg_strength, cfg_interval=self.cfg_interval,
                verbose=verbose,
            )
            x_0_pred = res.samples
            sample_pred.append(x_0_pred)

            src = sample_gt if kv_source == 'gt' else sample_pred
            denoised = src[frame_idx]
            pos_bank.append(model.build_kv_cache(
                x=denoised, frame_idx=frame_idx, cond=cond,
                t_noise=0.0, prev_kv_caches=pos_window))
            neg_bank.append(model.build_kv_cache(
                x=denoised, frame_idx=frame_idx, cond=neg_cond,
                t_noise=0.0, prev_kv_caches=neg_window))

        sample_gt_t = torch.cat(sample_gt, dim=0)
        sample_pred_t = torch.cat(sample_pred, dim=0)
        return {
            'sample_gt': sample_gt_t,
            'sample': sample_pred_t,
            'mse_per_frame': [
                F.mse_loss(sample_pred_t[i], sample_gt_t[i]).item()
                for i in range(num_frames)
            ],
        }

    # ------------------------------------------------------------------
    # Subclass hook for the shared run_snapshot.
    # The frame cap (`n`) is computed by the base ``_frame_count`` from
    # ``dataset.val_num_frames`` — no SS-specific override needed.
    # ------------------------------------------------------------------
    def _dispatch_inference(self, mode, scene, n, verbose=False):
        kv_source = 'gt' if mode == 'tf_kvcache' else 'self'
        out = self._run_inference_pass(
            scene, num_frames=n, kv_source=kv_source, verbose=verbose,
        )
        return {'mse': out['mse_per_frame']}


class DiffusionForcingSSImageConditionedTrainer(
    ImageConditionedMixin, DiffusionForcingSSTrainer
):
    """Image-conditioned variant — DINOv2 cond encoding + CFG dropout."""
    pass
