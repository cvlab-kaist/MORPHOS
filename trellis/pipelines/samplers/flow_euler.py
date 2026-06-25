from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class FlowEulerKVCacheGuidanceIntervalSampler(FlowEulerSampler):
    """Euler sampler for diffusion-forcing *streaming* inference with **dual**
    KV caches (full-window classifier-free guidance).

    Denoises one frame's latent against two parallel KV caches of
    previously-denoised frames via ``model.forward_with_kvcache``:

    - ``pos_kv_caches``: cached with the real ``cond`` (positive branch);
    - ``neg_kv_caches``: cached with ``neg_cond`` (negative branch).

    The entire negative path is unconditional — the current frame is denoised
    with ``neg_cond`` AND attends to the uncond cache — matching diffusion-
    forcing training that drops the cond on the whole window together.

    CFG uses the standard TRELLIS interval form (same as
    :class:`FlowEulerGuidanceIntervalSampler`)::

        v = (1 + cfg) * v_pos - cfg * v_neg      (t in cfg_interval)
        v = v_pos                                 (otherwise)

    Works for **both** stages:

    - SS: dense ``(1, C, R, R, R)`` latents — Euler step ``x_t - dt * v``;
    - SLat: ``SparseTensor`` latents — Euler step on ``.feats``.

    The stage is auto-detected from whether ``forward_with_kvcache`` returns an
    object with ``.feats`` (SparseTensor). Timesteps are pre-scaled by 1000
    (diff-forcing ``forward_with_kvcache`` does not rescale internally).
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,                          # dense tensor (SS) or SparseTensor (SLat)
        cond,
        neg_cond,
        frame_idx: int,
        pos_kv_caches,
        neg_kv_caches,
        steps: int = 25,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = False,
    ):
        """
        Args:
            model: diff-forcing model exposing ``forward_with_kvcache``.
            noise: initial noisy latent for the current frame (dense or sparse).
            cond / neg_cond: positive / negative conditioning.
            frame_idx: absolute frame index (logging / API compat).
            pos_kv_caches: oldest-first prev-frame caches built with ``cond``.
            neg_kv_caches: oldest-first prev-frame caches built with ``neg_cond``.
            steps / rescale_t / cfg_strength / cfg_interval: sampling controls.

        Returns:
            edict with ``samples`` (final latent) and ``pred_x_t``.
        """
        x_t = noise
        is_sparse = hasattr(x_t, 'feats')
        device = x_t.feats.device if is_sparse else x_t.device
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        ret = edict({"samples": None, "pred_x_t": []})
        for i in tqdm(range(steps), desc="Sampling", disable=not verbose):
            t = float(t_seq[i])
            dt = float(t_seq[i] - t_seq[i + 1])
            t_scaled = torch.tensor([t * 1000.0], device=device)

            out_pos = model.forward_with_kvcache(
                x_t, t_scaled, cond, frame_idx=frame_idx, kv_caches=pos_kv_caches,
            )
            v_pos = out_pos.feats if is_sparse else out_pos
            if cfg_interval[0] <= t <= cfg_interval[1]:
                out_neg = model.forward_with_kvcache(
                    x_t, t_scaled, neg_cond, frame_idx=frame_idx, kv_caches=neg_kv_caches,
                )
                v_neg = out_neg.feats if is_sparse else out_neg
                v = (1.0 + cfg_strength) * v_pos - cfg_strength * v_neg
            else:
                v = v_pos

            if is_sparse:
                x_t = x_t.replace(x_t.feats - dt * v)
            else:
                x_t = x_t - dt * v
            ret.pred_x_t.append(x_t)
        ret.samples = x_t
        return ret
