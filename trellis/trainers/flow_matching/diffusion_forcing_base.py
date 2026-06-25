"""Shared Diffusion-Forcing trainer base for SS and SLat.

Holds the bits that don't depend on the stage's latent type (dense SS vs
``SparseTensor`` SLat): sampler factory, per-frame ``t`` sampling, snapshot
orchestration (sharded across DDP ranks, mode-list driven, latent-MSE plus
subclass-emitted extras), and the standard ``__init__`` knobs.

Subclasses provide:
- ``_supported_modes``: tuple of snapshot mode names this stage can run.
- ``training_losses(...)``: stage-specific training step.
- ``_dispatch_inference(mode, scene, n, verbose) -> Dict[str, List[float]]``:
  run one (mode, scene) pair and return per-metric value lists. Required key
  ``'mse'``; any extra keys (e.g. ``'lpips'``, ``'clip'``, ``'dreamsim'``)
  are emitted at ``val_<metric>_<mode>`` in the snapshot output.
"""
from typing import *
import torch
import torch.nn.functional as F
import numpy as np

from ...pipelines import samplers
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin


class DiffusionForcingTrainerBase(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """Shared infrastructure for SS / SLat Diffusion-Forcing trainers."""

    DDP_FIND_UNUSED_PARAMETERS = False
    # Subclasses narrow this to the modes they actually implement.
    _supported_modes: Tuple[str, ...] = (
        "ar_kvcache", "tf_kvcache", "ar", "tf",
    )

    def __init__(
        self,
        *args,
        window_size: int = 3,
        independent_t: bool = True,
        cfg_strength: float = 3.0,
        snapshot_steps: int = 50,
        rescale_t: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.5, 1.0),
        snapshot_modes: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            window_size: Frames in the diff-forcing window (W).
            independent_t: True (the diff-forcing setting) draws an iid noise
                level per (scene, frame) slot; False shares one ``t`` across
                all W slots of each scene.
            cfg_strength: CFG scale at snapshot time. Form B sampler combine.
            snapshot_steps: Euler steps per frame at snapshot.
            rescale_t: Shifted-flow timestep rescale (TRELLIS pretrained 3.0).
            cfg_interval: Noise interval in which CFG is applied
                (TRELLIS pretrained (0.5, 1.0)). (0.0, 1.0) = every step.
            snapshot_modes: Subset of ``_supported_modes`` to run per val
                scene. Default ``["ar_kvcache"]``.
        """
        super().__init__(*args, **kwargs)
        self.window_size = int(window_size)
        self.independent_t = bool(independent_t)
        self.cfg_strength = float(cfg_strength)
        self.snapshot_steps = int(snapshot_steps)
        self.rescale_t = float(rescale_t)
        self.cfg_interval = tuple(cfg_interval)
        modes = list(snapshot_modes) if snapshot_modes else ["ar_kvcache"]
        bad = [m for m in modes if m not in self._supported_modes]
        if bad:
            raise ValueError(
                f"Unknown snapshot_modes: {bad}; "
                f"supported by this stage: {self._supported_modes}"
            )
        self.snapshot_modes = modes

    # ------------------------------------------------------------------
    # Shared sampler + t-sampling helpers
    # ------------------------------------------------------------------
    def get_sampler(self, **kwargs):
        """Dual cond/uncond KV-cache Euler sampler with rescale_t + cfg_interval."""
        return samplers.FlowEulerKVCacheGuidanceIntervalSampler(self.sigma_min)

    def _sample_t_per_frame(
        self, B: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Sample per-(scene, slot) noise level in [0, 1].

        ``independent_t=True`` (the diff-forcing setting): B·W iid draws from
        the parent's ``t_schedule`` (``self.sample_t``).
        ``independent_t=False``: B iid draws, broadcast to all W slots of
        the same scene.

        Returns:
            ``(B, W)`` float tensor on ``device``.
        """
        if self.independent_t:
            return self.sample_t(B * W).to(device).float().view(B, W)
        t = self.sample_t(B).to(device).float()
        return t.view(B, 1).expand(B, W).contiguous()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _enumerate_val_scenes(self):
        """Yield raw scene dicts for validation. Default delegates to dataset."""
        if hasattr(self.dataset, "get_all_val_scenes"):
            return list(self.dataset.get_all_val_scenes())
        if hasattr(self.dataset, "get_val_data"):
            return [self.dataset.get_val_data()]
        return [self.dataset[0]]

    def _frame_count(self, scene: Dict) -> int:
        """How many frames to run per snapshot for ``scene``."""
        n = scene["num_frames"]
        cap = getattr(self.dataset, "val_num_frames", None)
        return min(n, cap) if cap is not None else n

    def _dispatch_inference(
        self, mode: str, scene: Dict, n: int, verbose: bool = False,
    ) -> Dict[str, List[float]]:  # pragma: no cover - abstract
        """Run one snapshot mode on one scene; return per-metric value lists.

        At minimum, the returned dict must contain key ``'mse'`` with a list
        of per-frame latent MSEs (length ``n``). Subclasses may add extra
        keys (e.g. ``'lpips'``) with any number of values; each metric is
        averaged independently at the end.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Snapshot orchestration (shared)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size, verbose=False) -> Dict:
        """Per-mode metric means across all val scenes × frames.

        Output is only mode-aggregated scalars ``val_<metric>_<mode>``. No
        per-scene / per-frame breakdowns. DDP shards val scenes
        ``rank::world_size``; per-rank metric lists merge via
        ``dist.all_gather_object`` (the cross-rank sync point).
        """
        import torch.distributed as dist

        is_dist = dist.is_initialized() and self.world_size > 1
        rank = dist.get_rank() if is_dist else 0
        world_size = self.world_size if is_dist else 1

        accum: Dict[str, Dict[str, List[float]]] = {
            m: {} for m in self.snapshot_modes
        }
        val_scenes = self._enumerate_val_scenes()

        for idx, scene in enumerate(val_scenes):
            if idx % world_size != rank:
                continue
            n = self._frame_count(scene)
            for mode in self.snapshot_modes:
                per_metric = self._dispatch_inference(
                    mode, scene, n, verbose=verbose,
                )
                assert "mse" in per_metric, (
                    f"_dispatch_inference must return 'mse' for mode={mode}"
                )
                for k, vs in per_metric.items():
                    accum[mode].setdefault(k, []).extend(vs)
            torch.cuda.empty_cache()

        if is_dist:
            gathered: List[Optional[Dict]] = [None] * world_size
            dist.all_gather_object(gathered, accum)
            merged: Dict[str, Dict[str, List[float]]] = {
                m: {} for m in self.snapshot_modes
            }
            for part in gathered:
                if part is None:
                    continue
                for m in self.snapshot_modes:
                    for k, vs in part[m].items():
                        merged[m].setdefault(k, []).extend(vs)
            accum = merged

        output: Dict[str, Dict] = {}
        for mode in self.snapshot_modes:
            for metric, vals in accum[mode].items():
                if vals:
                    output[f"val_{metric}_{mode}"] = {
                        "value": float(np.mean(vals)),
                        "type": "scalar",
                    }
        return output
