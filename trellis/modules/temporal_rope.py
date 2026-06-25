"""Unified 1D rotary position embedding (RoPE) for SS and SLat temporal models.

Both stages encode per-frame position by rotating Q/K through ``Rope1D`` at
attention time. RoPE has no learnable parameters; only the inverse-frequency
buffer ``freqs = 1/base^(2i/head_dim)`` is held by the module.

Shape contract: ``x`` is ``(..., L, H, D)`` with even ``D``; ``idx`` is any
shape that broadcasts against ``x``'s leading-then-``L`` axes. The same call
covers SS's batched ``(B, T, H, d)`` with per-frame ``(T,)`` positions, the
SS KV-cache path's ``(B, L, H, d)`` with broadcast scalar ``(1,)`` positions,
and SLat's flat ``(N, H, D)`` with per-token ``(N,)`` positions.

Compute path is fp32 for cos/sin then casts back to ``x.dtype`` for the
multiply-add — matching the SS implementation it replaces. The numerical
difference vs SLat v3's prior complex-multiplication path (also fp32) is
sub-1ulp on the output and below noise in downstream metrics.
"""
from typing import Optional
import torch
import torch.nn as nn


class Rope1D(nn.Module):
    """1D RoPE applied per-head to Q or K.

    Args:
        head_dim: Per-head channel count. Must be even.
        base: RoPE inverse-freq base (default 10000.0, the canonical value).
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Rotate Q or K.

        Args:
            x: ``(..., L, H, D)`` tensor.
            idx: shape that broadcasts against ``x``'s axes up to and including
                 ``L``. Typical shapes: ``(L,)``, ``(B, L)``, ``(1,)``.

        Returns:
            Rotated tensor with the same shape and dtype as ``x``.
        """
        freqs = self.freqs.to(x.device)
        # angles: (..., L, half)  --  unsqueeze pairs the head dim later.
        angles = idx.to(freqs.dtype).unsqueeze(-1) * freqs
        # cos/sin: (..., L, 1, half) — broadcasts over H.
        cos = torch.cos(angles).unsqueeze(-2).to(x.dtype)
        sin = torch.sin(angles).unsqueeze(-2).to(x.dtype)
        x_ev = x[..., 0::2]
        x_od = x[..., 1::2]
        out_ev = x_ev * cos - x_od * sin
        out_od = x_ev * sin + x_od * cos
        return torch.stack([out_ev, out_od], dim=-1).flatten(-2)
