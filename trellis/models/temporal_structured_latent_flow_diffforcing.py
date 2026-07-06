"""
Diffusion Forcing SLat flow model (v3).

Notable design choices:
- No noise level embedder. adaLN alone carries the per-frame noise condition.
- Per-frame position embedding applied to Q and K inside self-attention
  (after qk_rms_norm) via 1D RoPE — parameter-free and shared with the SS
  diffforcing model (``trellis.modules.temporal_rope.Rope1D``).
- Window-local frame indices ``[0, 1, ..., W-1]`` where 0 = oldest in window,
  W-1 = current frame. Indices are derived from list ordering in training,
  and from ``len(prev_kv_caches)`` at inference.
- KV cache stores **pre-temporal-embedding** K (raw post-to_qkv + qk_rms_norm).
  Temporal embedding is re-applied on every attention call using fresh
  window-local slots, so sliding the window does not invalidate cached K.
"""
from typing import *
from contextlib import contextmanager, nullcontext
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.sparse.basic import SparseTensor
from ..modules.sparse import ATTN
from ..modules.temporal_rope import Rope1D
from ..utils.elastic_utils import ElasticModuleMixin
from .structured_latent_flow import ElasticSLatFlowModel

_xops = None
_flash_attn = None
_flex_attn_compiled = None

def _attn_call(q, k, v):
    global _xops, _flash_attn
    if ATTN == 'xformers':
        if _xops is None:
            import xformers.ops as m; _xops = m
        return _xops.memory_efficient_attention(q, k, v)[0]
    elif ATTN == 'flash_attn':
        if _flash_attn is None:
            import flash_attn as m; _flash_attn = m
        return _flash_attn.flash_attn_func(q, k, v)[0]
    elif ATTN == 'flex_attn':
        # flex_attention is exposed separately via the model's block-causal
        # path (`_frame_causal_self_attn`). Single-block attention (no causal
        # mask needed) like `forward_with_kvcache` falls back to flash_attn,
        # which is already correct because we feed the full KV slice and
        # don't need block-causal there.
        if _flash_attn is None:
            import flash_attn as m; _flash_attn = m
        return _flash_attn.flash_attn_func(q, k, v)[0]
    else:
        raise ValueError(f"Unknown ATTN: {ATTN}")


def _get_flex_attn():
    """Lazily import + torch.compile FlexAttention once; cache on first call.
    Uses `dynamic=True` so the same compiled artifact handles the varying
    `total_tokens` across scenes (different sparse layouts + coord_drop).
    """
    global _flex_attn_compiled
    if _flex_attn_compiled is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex_attn_compiled = torch.compile(flex_attention, dynamic=True)
    return _flex_attn_compiled


class DiffusionForcingTemporalSlatFlowModel(ElasticModuleMixin, nn.Module):
    """
    Diffusion Forcing SLat flow: symmetric W-frame denoising; window-local temporal
    position applied inside self-attention (Q, K). KV cache holds raw K.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        window_size: int = 3,
        trainable_scope: str = "self_attn",
        pretrained_base: str = 'microsoft/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16',
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.trainable_scope = trainable_scope
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.base_model = ElasticSLatFlowModel(
            resolution=resolution, in_channels=in_channels,
            model_channels=model_channels, cond_channels=cond_channels,
            out_channels=out_channels, num_blocks=num_blocks,
            num_heads=num_heads, num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio, patch_size=patch_size,
            num_io_res_blocks=num_io_res_blocks,
            io_block_channels=io_block_channels, pe_mode=pe_mode,
            use_fp16=use_fp16, use_checkpoint=use_checkpoint,
            use_skip_connection=use_skip_connection, share_mod=share_mod,
            qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
        )

        if pretrained_base is not None:
            self._load_pretrained_base(pretrained_base)

        self.base_model.requires_grad_(False)
        self._unfreeze_scope(trainable_scope)

        # Resolve actual head_dim used inside the base blocks; 1D RoPE on Q/K
        # (shared module with the SS diffforcing model).
        resolved_num_heads = self.base_model.blocks[0].self_attn.num_heads
        head_dim = model_channels // resolved_num_heads
        self.temporal_embedder = Rope1D(head_dim)

        self._log_trainable_params()

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _unfreeze_scope(self, scope):
        """``adaLN_modulation`` is always unfrozen *except* when
        ``scope="none"``; frozen adaLN gates throttle gradients into the rest
        of the block by their tiny pretrained values, so any scope that wants
        to update anything in the transformer should also let the gates grow.
        """
        if scope == "none":
            return
        if scope == "self_attn":
            for block in self.base_model.blocks:
                block.self_attn.requires_grad_(True)
        elif scope == "transformer":
            for block in self.base_model.blocks:
                block.requires_grad_(True)
        elif scope == "all":
            self.base_model.requires_grad_(True)
        else:
            raise ValueError(f"Unknown trainable_scope: {scope}")

        # adaLN_modulation is always trainable except scope='none'.
        for block in self.base_model.blocks:
            if hasattr(block, 'adaLN_modulation'):
                block.adaLN_modulation.requires_grad_(True)

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[DiffusionForcingSLatFlow] Parameters: {total/1e6:.1f}M total, '
              f'{trainable/1e6:.1f}M trainable, {(total-trainable)/1e6:.1f}M frozen '
              f'(scope={self.trainable_scope}, window_size={self.window_size})')

    def _load_pretrained_base(self, pretrained_path):
        import os
        is_local = os.path.exists(f"{pretrained_path}.json") and os.path.exists(f"{pretrained_path}.safetensors")
        if is_local:
            model_file = f"{pretrained_path}.safetensors"
        else:
            from huggingface_hub import hf_hub_download
            parts = pretrained_path.split('/')
            repo_id = f'{parts[0]}/{parts[1]}'
            model_name = '/'.join(parts[2:])
            model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")
        from safetensors.torch import load_file
        self.base_model.load_state_dict(load_file(model_file))
        print(f'[DiffusionForcingSLatFlow] Loaded pretrained base from {pretrained_path}')

    def convert_to_fp16(self):
        self.base_model.convert_to_fp16()

    def convert_to_fp32(self):
        self.base_model.convert_to_fp32()

    def set_gradient_checkpointing(self, enable=True):
        for block in self.base_model.blocks:
            block.use_checkpoint = enable

    def _get_input_size(self, x, *args, **kwargs):
        return x.feats.shape[0] if isinstance(x, SparseTensor) else x.shape[0]

    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        nb = len(self.base_model.blocks)
        nc = min(math.ceil((1 - mem_ratio) * nb) + 1, nb)
        exact = 1 - (nc - 1) / nb
        for i in range(nb):
            self.base_model.blocks[i].use_checkpoint = i < nc
        yield exact
        for i in range(nb):
            self.base_model.blocks[i].use_checkpoint = False

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ------------------------------------------------------------------
    # Per-frame input path
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_frame(self, frame_st: SparseTensor, t_emb: torch.Tensor):
        base = self.base_model
        h = base.input_layer(frame_st).type(self.dtype)
        skips = []
        for blk in base.input_blocks:
            h = blk(h, t_emb)
            skips.append(h.feats)
        if base.pe_mode == "ape":
            h = h + base.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        return h.replace(h.feats.detach()), [s.detach() for s in skips]

    # ------------------------------------------------------------------
    # Frame-causal self-attention (training)
    # ------------------------------------------------------------------
    def _frame_causal_self_attn(self, self_attn, h, frame_ends, per_token_fidx):
        qkv = self_attn._linear(self_attn.to_qkv, h)
        qkv = self_attn._fused_pre(qkv, num_fused=3)
        if self_attn.qk_rms_norm:
            q_st, k_st, v_st = qkv.unbind(dim=1)
            q_st = self_attn.q_rms_norm(q_st)
            k_st = self_attn.k_rms_norm(k_st)
            q_raw, k_raw, v_raw = q_st.feats, k_st.feats, v_st.feats
        else:
            q_raw, k_raw, v_raw = qkv.feats.unbind(dim=1)

        # Temporal embedding on Q and K (V unchanged). Applied to the full
        # concatenated token stream before any per-frame work.
        q_raw = self.temporal_embedder(q_raw, per_token_fidx)
        k_raw = self.temporal_embedder(k_raw, per_token_fidx)

        if ATTN == 'flex_attn':
            # FlexAttention path: one fused kernel over the packed sequence
            # with a block-causal mask derived from `per_token_fidx`.
            # `per_token_fidx[i]` is the window-local frame index of token i;
            # token i attends to token j iff fidx[j] <= fidx[i]. This makes
            # the mask independent of per-frame token counts (handles
            # variable-N-per-frame naturally, including post-coord_drop).
            from torch.nn.attention.flex_attention import create_block_mask
            flex_attn = _get_flex_attn()
            fidx = per_token_fidx
            N = q_raw.shape[0]

            def block_causal_mod(b, h_, q_idx, kv_idx):
                return fidx[q_idx] >= fidx[kv_idx]

            block_mask = create_block_mask(
                block_causal_mod, B=None, H=None, Q_LEN=N, KV_LEN=N,
                device=q_raw.device,
            )
            # (N, H, D) -> (1, H, N, D) for FlexAttention.
            q = q_raw.transpose(0, 1).unsqueeze(0)
            k = k_raw.transpose(0, 1).unsqueeze(0)
            v = v_raw.transpose(0, 1).unsqueeze(0)
            out = flex_attn(q, k, v, block_mask=block_mask)
            # (1, H, N, D) -> (N, H, D) -> (N, H*D)
            out = out.squeeze(0).transpose(0, 1).contiguous()
            out = out.reshape(-1, q_raw.shape[-1] * self_attn.num_heads)
        else:
            # Default path: per-frame Python loop. Each frame's queries
            # attend to all keys/values from frames 0..i (block-causal).
            frame_starts = [0] + frame_ends[:-1]
            out_parts = []
            for i in range(len(frame_ends)):
                q_i = q_raw[frame_starts[i]:frame_ends[i]]
                k_i = k_raw[:frame_ends[i]]
                v_i = v_raw[:frame_ends[i]]
                out_i = _attn_call(q_i.unsqueeze(0), k_i.unsqueeze(0), v_i.unsqueeze(0))
                out_parts.append(out_i)
            out = torch.cat(out_parts, dim=0).reshape(-1, q_raw.shape[-1] * self_attn.num_heads)

        h_out = h.replace(out)
        h_out = self_attn._linear(self_attn.to_out, h_out)
        return h_out

    # ------------------------------------------------------------------
    # KV-cache self-attention (inference). Cache holds pre-temporal-emb K.
    # ------------------------------------------------------------------
    def _kvcache_self_attn(
        self, self_attn, h_curr,
        kv_cache_k_raw, kv_cache_v,
        cached_fidx, curr_fidx,
    ):
        """Q from current frame, K/V from [cached_raw_prev + current_raw].
        Temporal embedding re-applied on Q and the concatenated K using the
        current window slots. Returns raw (pre-embedding) K_curr for caching.
        """
        qkv = self_attn._linear(self_attn.to_qkv, h_curr)
        qkv = self_attn._fused_pre(qkv, num_fused=3)
        if self_attn.qk_rms_norm:
            q_st, k_st, v_st = qkv.unbind(dim=1)
            q_st = self_attn.q_rms_norm(q_st)
            k_st = self_attn.k_rms_norm(k_st)
            q_raw, k_curr_raw, v_curr = q_st.feats, k_st.feats, v_st.feats
        else:
            q_raw, k_curr_raw, v_curr = qkv.feats.unbind(dim=1)

        if kv_cache_k_raw is not None:
            k_all_raw = torch.cat([kv_cache_k_raw, k_curr_raw], dim=0)
            v_all = torch.cat([kv_cache_v, v_curr], dim=0)
            k_all_fidx = torch.cat([cached_fidx, curr_fidx], dim=0)
        else:
            k_all_raw = k_curr_raw
            v_all = v_curr
            k_all_fidx = curr_fidx

        q = self.temporal_embedder(q_raw, curr_fidx)
        k_all = self.temporal_embedder(k_all_raw, k_all_fidx)

        out = _attn_call(q.unsqueeze(0), k_all.unsqueeze(0), v_all.unsqueeze(0))
        out = out.reshape(q.shape[0], -1)
        h_out = h_curr.replace(out)
        h_out = self_attn._linear(self_attn.to_out, h_out)
        return h_out, k_curr_raw.detach(), v_curr.detach()

    # ------------------------------------------------------------------
    # Per-block forward (training: causal self-attn, batched cross-attn)
    # ------------------------------------------------------------------
    # Self-attn:  v3 causal implementation — uses per_token_fidx + frame_ends.
    # Cross-attn: original block.cross_attn(h, cond_packed) on B=F sparse h
    #             and dense cond_packed [F, L, C] (per-frame).
    # FFN/norms:  unchanged (block.mlp / block.norm{1,2,3}).
    # AdaLN:      per-frame t_emb_packed [F, 6C] -> per-token via fidx index.
    def _forward_block_causal(self, block, x, t_emb_packed, cond_packed,
                              per_token_fidx, frame_ends):
        # Per-frame -> per-token modulation (frame-wise adaLN)
        if block.share_mod:
            mod_per_frame = t_emb_packed                       # [F, 6C]
        else:
            mod_per_frame = block.adaLN_modulation(t_emb_packed)  # [F, 6C]
        mod_per_token = mod_per_frame[per_token_fidx]          # [N_total, 6C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod_per_token.chunk(6, dim=1)

        # Self-attn (causal, current implementation, unchanged)
        h = x.replace(block.norm1(x.feats))
        h = h.replace(h.feats * (1 + scale_msa) + shift_msa)
        h = self._frame_causal_self_attn(block.self_attn, h, frame_ends, per_token_fidx)
        h = h.replace(h.feats * gate_msa)
        x = x.replace(x.feats + h.feats)

        # Cross-attn (original code path: block.cross_attn natively batches over F)
        h = x.replace(block.norm2(x.feats))
        h = block.cross_attn(h, cond_packed)
        x = x.replace(x.feats + h.feats)

        # FFN (original code, unchanged)
        h = x.replace(block.norm3(x.feats))
        h = h.replace(h.feats * (1 + scale_mlp) + shift_mlp)
        h = block.mlp(h)
        h = h.replace(h.feats * gate_mlp)
        x = x.replace(x.feats + h.feats)
        return x

    # ------------------------------------------------------------------
    # Per-block forward (inference: KV cache, single frame)
    # ------------------------------------------------------------------
    def _forward_block_kvcache(
        self, block, x, t_emb, cond,
        kv_cache_k_raw, kv_cache_v,
        cached_fidx, curr_fidx,
    ):
        """Single-frame block with KV cache from prev frames.
        Returns (x_out, new_k_raw, new_v) for caching."""
        mod = t_emb if block.share_mod else block.adaLN_modulation(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)

        h = x.replace(block.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h, new_k_raw, new_v = self._kvcache_self_attn(
            block.self_attn, h,
            kv_cache_k_raw, kv_cache_v,
            cached_fidx, curr_fidx,
        )
        h = h * gate_msa
        x = x + h

        h = x.replace(block.norm2(x.feats))
        h = block.cross_attn(h, cond)
        x = x + h

        h = x.replace(block.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = block.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x, new_k_raw, new_v

    # ------------------------------------------------------------------
    # Output path (unchanged from v1)
    # ------------------------------------------------------------------
    def _output_path(self, h_f, skips_f, t_emb, out_dtype):
        base = self.base_model
        for blk, skip in zip(base.out_blocks, reversed(skips_f)):
            if base.use_skip_connection:
                h_f = blk(h_f.replace(torch.cat([h_f.feats, skip], dim=1)), t_emb)
            else:
                h_f = blk(h_f, t_emb)
        h_f = h_f.replace(F.layer_norm(h_f.feats, h_f.feats.shape[-1:]))
        h_f = base.out_layer(h_f.type(out_dtype))
        return h_f.feats

    # ------------------------------------------------------------------
    # Derive window-local slots from a list of cached frame K tensors.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_cached_fidx(prev_kv_caches_per_frame, blk_idx: int, device) -> torch.Tensor:
        """Given list of per-frame caches (oldest->newest), return [N_cached]
        long tensor whose j-th block of tokens carries the window-local slot
        of that cached frame (slot j = 0 for oldest)."""
        if not prev_kv_caches_per_frame:
            return torch.empty(0, dtype=torch.long, device=device)
        parts = []
        for slot, cache in enumerate(prev_kv_caches_per_frame):
            n_tokens = cache[blk_idx][0].shape[0]
            parts.append(torch.full((n_tokens,), slot, dtype=torch.long, device=device))
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Encode a clean frame and cache its K/V at every block
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_kv_cache(
        self, frame_st: SparseTensor, t_noise: float,
        frame_idx: int, cond: torch.Tensor,
        sigma_min: float = 1e-5,
        prev_kv_caches: Optional[List[List[tuple]]] = None,
    ):
        """Encode a frame and run it through all blocks to collect per-block K/V.

        The `frame_idx` argument is accepted for drop-in compatibility with v1
        callers, but is **ignored**. Window-local slots are derived from the
        length of `prev_kv_caches`: oldest cached frame -> slot 0, new current
        frame -> slot `len(prev_kv_caches)`.

        The cached K returned is RAW (post to_qkv + qk_rms_norm, before
        temporal embedding) so it remains valid when the window slides.
        """
        base = self.base_model
        device = frame_st.device

        if t_noise > 0:
            noise = torch.randn_like(frame_st.feats)
            noised_feats = (1 - t_noise) * frame_st.feats + (sigma_min + (1 - sigma_min) * t_noise) * noise
            frame_st = frame_st.replace(noised_feats)

        t_emb = base.t_embedder(torch.tensor([t_noise * 1000], device=device))
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        h, skips = self._encode_frame(frame_st, t_emb)

        n_prev = len(prev_kv_caches) if prev_kv_caches else 0
        curr_slot = n_prev
        N_curr = h.feats.shape[0]
        curr_fidx = torch.full((N_curr,), curr_slot, dtype=torch.long, device=device)

        kv_cache = []
        for blk_idx, blk in enumerate(base.blocks):
            if prev_kv_caches:
                cached_k = torch.cat([c[blk_idx][0] for c in prev_kv_caches], dim=0)
                cached_v = torch.cat([c[blk_idx][1] for c in prev_kv_caches], dim=0)
                cached_fidx = self._build_cached_fidx(prev_kv_caches, blk_idx, device)
            else:
                cached_k, cached_v = None, None
                cached_fidx = torch.empty(0, dtype=torch.long, device=device)
            h, new_k_raw, new_v = self._forward_block_kvcache(
                blk, h, t_emb, cond,
                cached_k, cached_v, cached_fidx, curr_fidx,
            )
            kv_cache.append((new_k_raw, new_v))
        return kv_cache

    # ------------------------------------------------------------------
    # Single-frame denoising with KV cache (for sampler)
    # ------------------------------------------------------------------
    def forward_with_kvcache(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        frame_idx: int,
        kv_caches: Optional[List[List[tuple]]] = None,
    ) -> SparseTensor:
        """Denoise one frame using cached K/V from previous frames.

        `frame_idx` is accepted but ignored; window-local slots derived from
        `len(kv_caches)` so sliding-window rollouts work without caller change.
        """
        base = self.base_model
        device = x.device

        t_emb = base.t_embedder(t)
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        h, skips = self._encode_frame(x, t_emb)

        n_prev = len(kv_caches) if kv_caches else 0
        curr_slot = n_prev
        N_curr = h.feats.shape[0]
        curr_fidx = torch.full((N_curr,), curr_slot, dtype=torch.long, device=device)

        for blk_idx, blk in enumerate(base.blocks):
            if kv_caches:
                cached_k = torch.cat([c[blk_idx][0] for c in kv_caches], dim=0)
                cached_v = torch.cat([c[blk_idx][1] for c in kv_caches], dim=0)
                cached_fidx = self._build_cached_fidx(kv_caches, blk_idx, device)
            else:
                cached_k, cached_v = None, None
                cached_fidx = torch.empty(0, dtype=torch.long, device=device)
            h, _, _ = self._forward_block_kvcache(
                blk, h, t_emb, cond,
                cached_k, cached_v, cached_fidx, curr_fidx,
            )

        out_feats = self._output_path(h, skips, t_emb, x.dtype)
        return h.replace(out_feats)

    # ------------------------------------------------------------------
    # Training forward (list of frames, causal)
    # ------------------------------------------------------------------
    def forward(
        self,
        x_frames_or_x,
        t_per_frame_or_t,
        cond: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
        per_frame_cond: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> SparseTensor:
        if not isinstance(x_frames_or_x, list):
            raise ValueError(
                "DiffusionForcingTemporalSlatFlowModel.forward() expects a list of SparseTensors "
                "(training mode). For inference, use forward_with_kvcache()."
            )
        return self._forward_training(
            x_frames_or_x, t_per_frame_or_t, cond,
            frame_indices, per_frame_cond,
        )

    def _forward_training(self, x_frames, t_per_frame, cond,
                          frame_indices, per_frame_cond):
        """Training forward with frames packed in the batch dimension.

        Input/output paths and cross-attention reuse the original simple SLat
        code (`SLatFlowModel.forward`) by setting `coords[:, 0] = frame_idx`
        and passing per-frame `emb` of shape [F, C] / per-frame `cond` of
        shape [F, L, C]. spconv dispatches per batch (per frame) natively.

        Self-attention is the only operator that needs cross-frame mixing —
        it consumes the same packed SparseTensor but uses `per_token_fidx`
        (= h.coords[:, 0]) for causal slicing and temporal RoPE. AdaLN is
        applied frame-wise via `mod_per_frame[per_token_fidx]` indexing.

        The caller-supplied `frame_indices` is ignored: window-local indexing
        (oldest=0, current=W-1) is induced by the order of `x_frames`.
        """
        base = self.base_model
        F_frames = len(x_frames)
        device = x_frames[0].device

        # 1) Pack frames as one B=F SparseTensor (coords[:, 0] = frame_idx)
        coords_list, feats_list = [], []
        for i, frame_st in enumerate(x_frames):
            c = frame_st.coords.clone()
            c[:, 0] = i
            coords_list.append(c)
            feats_list.append(frame_st.feats)
        x_packed = SparseTensor(
            coords=torch.cat(coords_list),
            feats=torch.cat(feats_list),
        )
        x_packed._shape = torch.Size([F_frames, *x_frames[0].feats.shape[1:]])

        # 2) Per-frame t_emb [F, C] (frame-wise adaLN noise condition)
        t_emb_packed = base.t_embedder(t_per_frame.float() * 1000)   # [F, C]
        if base.share_mod:
            t_emb_packed = base.adaLN_modulation(t_emb_packed)
        t_emb_packed = t_emb_packed.type(self.dtype)

        # 3) Per-frame DINOv2 cond [F, L, C]
        if per_frame_cond is None:
            cond_packed = cond.expand(F_frames, -1, -1).type(self.dtype)
        else:
            cond_packed = torch.cat(per_frame_cond, dim=0).type(self.dtype)

        # 4) Input path — original SLatFlowModel.forward code (lines 240-255),
        #    runs once on B=F. spconv batches per coords[:, 0]; SparseResBlock3d
        #    broadcasts emb [F, C] over per-batch tokens.
        #
        #    By default the encoder side (input_layer, input_blocks, pos_embedder)
        #    is held at its pretrained weights: we run it under torch.no_grad()
        #    and detach its outputs, so no gradient ever flows back into those
        #    params. With trainable_scope='all' the user explicitly opts into
        #    full fine-tuning, so we drop both the no_grad and the detach and
        #    let gradients flow through the encoder. (Costs more activation
        #    memory; can OOM at high batch.)
        encoder_trainable = (self.trainable_scope == 'all')
        encoder_ctx = nullcontext() if encoder_trainable else torch.no_grad()
        with encoder_ctx:
            h = base.input_layer(x_packed).type(self.dtype)
            skips = []
            for blk in base.input_blocks:
                h = blk(h, t_emb_packed)
                skips.append(h.feats)
            if base.pe_mode == "ape":
                h = h + base.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        if not encoder_trainable:
            h = h.replace(h.feats.detach())
            skips = [s.detach() for s in skips]

        # 5) Per-token frame index from coords[:, 0]; frame_ends from counts.
        #    spconv keeps batches contiguous, so cumulative counts still
        #    delimit per-frame token blocks for self-attn slicing.
        per_token_fidx = h.coords[:, 0].long()
        frame_token_counts = [
            int((per_token_fidx == i).sum().item()) for i in range(F_frames)
        ]
        frame_ends = list(np.cumsum(frame_token_counts))

        # 6) Transformer blocks: self-attn causal + cross-attn batched + FFN
        for blk in base.blocks:
            if blk.use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(
                    self._forward_block_causal,
                    blk, h, t_emb_packed, cond_packed, per_token_fidx, frame_ends,
                    use_reentrant=False,
                )
            else:
                h = self._forward_block_causal(
                    blk, h, t_emb_packed, cond_packed, per_token_fidx, frame_ends,
                )

        # 7) Output path — original SLatFlowModel.forward code (lines 259-267),
        #    runs once on B=F. Skips are batch-aligned with h (same coords[:,0]
        #    grouping) so the cat-with-skip works per frame.
        out_dtype = x_frames[0].dtype
        for blk, skip in zip(base.out_blocks, reversed(skips)):
            if base.use_skip_connection:
                h = blk(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb_packed)
            else:
                h = blk(h, t_emb_packed)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = base.out_layer(h.type(out_dtype))
        return h
