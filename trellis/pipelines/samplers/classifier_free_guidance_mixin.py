from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.

    When `cfg_neg_kv_caches` is provided, the negative branch uses that
    (typically uncond-built) KV cache instead of the positive branch's
    `kv_caches`. This matches diffusion-forcing training schemes that drop
    cond on the entire window together (full-window CFG): inference's
    negative branch is then `(neg_cond, uncond KV cache)` rather than
    `(neg_cond, cond KV cache)`. If `cfg_neg_kv_caches is None`, behavior is
    unchanged from the original single-cache CFG.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength,
                         cfg_neg_kv_caches=None, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        if cfg_neg_kv_caches is not None:
            neg_kwargs = {**kwargs, 'kv_caches': cfg_neg_kv_caches}
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **neg_kwargs)
        else:
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
