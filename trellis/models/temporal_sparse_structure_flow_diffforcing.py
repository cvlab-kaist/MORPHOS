"""
Diffusion Forcing Sparse-Structure flow model (stage 1).

Window of `window_size` frames processed jointly with **block-causal**
self-attention. Causality is realized by **per-frame slicing** of K/V (frame
i's Q attends to K/V of frames 0..i), NOT by a bool attention mask — so the
hot path can use `flash_attn.flash_attn_func` end-to-end. Each frame has an
independent noise level (diffusion forcing). Per-frame adaLN modulation and
per-frame DINOv2 cross-attention.

Training: list or tensor of W frames noised independently; forward returns
  per-frame velocity predictions. W flash_attn calls per block.
Inference: autoregressive — generate one frame at a time via
  `forward_with_kvcache`; after each frame is fully denoised call
  `build_kv_cache` to cache per-block (k, v) for reuse as context. One
  flash_attn call per block (Q from current, K/V from cached + current).
"""
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.spatial import patchify, unpatchify
from ..modules.temporal_rope import Rope1D
from .sparse_structure_flow import SparseStructureFlowModel


# Lazy flash_attn import — keeps the module importable in envs without
# flash_attn (e.g., for offline ckpt inspection) and avoids hard-failing on
# import. flash_attn is required at training/inference time on the GPU path.
_flash_attn = None


def _get_flash_attn():
    global _flash_attn
    if _flash_attn is None:
        import flash_attn as _m
        _flash_attn = _m
    return _flash_attn


class DiffusionForcingTemporalSSFlowModel(nn.Module):
    """
    Stage-1 diffusion-forcing model wrapping a pretrained SparseStructureFlowModel.

    Args:
        window_size: Max number of frames attended jointly (training) / in the
            sliding KV cache (inference). Default 3.
        trainable_scope: Which base-model params to unfreeze.
            - "self_attn": per-block self_attn + adaLN_modulation
            - "transformer": all transformer-block params
            - "all": entire base model
            - "none": everything frozen (base model only serves embeddings)
        pretrained_base: Path/HF id of the pretrained base SS flow checkpoint.
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
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        window_size: int = 3,
        trainable_scope: str = "self_attn",
        pretrained_base: str = 'microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16',
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.trainable_scope = trainable_scope
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.base_model = SparseStructureFlowModel(
            resolution=resolution,
            in_channels=in_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            share_mod=share_mod,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        if pretrained_base is not None:
            self._load_pretrained_base(pretrained_base)
        self.base_model.requires_grad_(False)
        self._unfreeze_scope(trainable_scope)

        # Frame-position info is carried via RoPE (Rotary Position Embedding)
        # on Q/K inside self-attention — parameter-free, multiplicative, and
        # preserves pretrained behavior at frame position 0. Shared module
        # with the SLat v3 model (trellis.modules.temporal_rope.Rope1D).
        head_dim = self.base_model.blocks[0].self_attn.head_dim
        self.rope = Rope1D(head_dim)

        self._log_trainable_params()

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _unfreeze_scope(self, scope: str):
        """Mirror of :meth:`DiffusionForcingTemporalSlatFlowModel._unfreeze_scope`
        so the two stages take the same set of trainable parameters per scope.

        ``adaLN_modulation`` is always unfrozen *except* when ``scope="none"``;
        frozen adaLN gates throttle gradients into the rest of the block by
        their tiny pretrained values, so any scope that wants to update
        anything in the transformer should also let the gates grow.
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
        print(
            f'[DiffusionForcingSSFlow] Parameters: {total/1e6:.1f}M total, '
            f'{trainable/1e6:.1f}M trainable, {(total-trainable)/1e6:.1f}M frozen '
            f'(scope={self.trainable_scope}, window_size={self.window_size})'
        )

    def _load_pretrained_base(self, pretrained_path: str):
        import os
        is_local = (
            os.path.exists(f"{pretrained_path}.json")
            and os.path.exists(f"{pretrained_path}.safetensors")
        )
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
        print(f'[DiffusionForcingSSFlow] Loaded pretrained base from {pretrained_path}')

    def convert_to_fp16(self):
        self.base_model.convert_to_fp16()

    def convert_to_fp32(self):
        self.base_model.convert_to_fp32()

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    # ------------------------------------------------------------------
    # Attention helpers (flash_attn end-to-end; bypass TRELLIS wrapper)
    # ------------------------------------------------------------------
    def _frame_causal_self_attn(
        self,
        self_attn,
        x: torch.Tensor,
        F_frames: int,
        L_per_frame: int,
    ) -> torch.Tensor:
        """Block-causal self-attention via per-frame `flash_attn_func` calls.

        Causality is realized by **slicing** rather than by a mask: for
        frame i in [0, F), Q comes from frame i and K/V come from frames
        0..i (concatenated). flash_attn doesn't accept arbitrary bool masks,
        so this is the only way to keep block-causal semantics on the FA path.

        Reuses the MultiHeadAttention's parameters (to_qkv, q/k_rms_norm,
        to_out) but bypasses its forward to inject RoPE on Q/K.
        """
        fa = _get_flash_attn()
        B, T, C = x.shape
        H = self_attn.num_heads
        d = self_attn.head_dim
        assert T == F_frames * L_per_frame, f"{T} != {F_frames}*{L_per_frame}"
        qkv = self_attn.to_qkv(x).reshape(B, T, 3, H, d)
        q, k, v = qkv.unbind(dim=2)
        if self_attn.qk_rms_norm:
            q = self_attn.q_rms_norm(q)
            k = self_attn.k_rms_norm(k)
        # RoPE: window position per token. All L tokens in frame f share pos=f.
        pos = torch.arange(F_frames, device=x.device, dtype=torch.long).repeat_interleave(L_per_frame)
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        # Per-frame slicing: q_i attends to k[:i+1], v[:i+1].
        q = q.view(B, F_frames, L_per_frame, H, d)
        k = k.view(B, F_frames, L_per_frame, H, d)
        v = v.view(B, F_frames, L_per_frame, H, d)
        outs = []
        for i in range(F_frames):
            q_i = q[:, i].contiguous()                                    # (B, L, H, d)
            k_i = k[:, :i + 1].reshape(B, (i + 1) * L_per_frame, H, d).contiguous()
            v_i = v[:, :i + 1].reshape(B, (i + 1) * L_per_frame, H, d).contiguous()
            outs.append(fa.flash_attn_func(q_i, k_i, v_i))                # (B, L, H, d)
        out = torch.stack(outs, dim=1).reshape(B, T, C)
        return self_attn.to_out(out)

    def _kvcache_self_attn(
        self,
        self_attn,
        x: torch.Tensor,
        cached_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q from current frame; K/V = [cached | cur]. Frame position is
        injected multiplicatively at attention time via RoPE based on WINDOW
        position (cached frame i -> window pos i, current -> window pos
        len(cached)) — so a cached entry stays valid as the sliding window
        advances.

        Cache format: each entry is `(K_pre_rms, V_bare)` where
          - K_pre_rms is `to_qkv(h)[K]` — no RMS-norm, no RoPE applied
          - V_bare   is `to_qkv(h)[V]` — unchanged at attention time (V is not
            rotated under RoPE).

        Returns (out, k_cur_pre_rms, v_cur_bare). Caller caches the two
        detached tensors verbatim.
        """
        fa = _get_flash_attn()
        B, L, C = x.shape
        H = self_attn.num_heads
        d = self_attn.head_dim
        qkv = self_attn.to_qkv(x).reshape(B, L, 3, H, d)
        q, k_cur_pre_rms, v_cur_bare = qkv.unbind(dim=2)

        num_cached = len(cached_list)
        cur_pos = num_cached

        K_parts = []
        V_parts = []
        # Cached: apply RMS-norm first (if configured), then RoPE rotation by
        # the cached frame's window position p.
        for p, (k_pre, v_bare) in enumerate(cached_list):
            k_p = k_pre.to(q.dtype)
            if self_attn.qk_rms_norm:
                k_p = self_attn.k_rms_norm(k_p)
            pos_p = torch.tensor([p], device=x.device, dtype=torch.long)
            k_p = self.rope(k_p, pos_p)
            K_parts.append(k_p)
            V_parts.append(v_bare.to(v_cur_bare.dtype))

        # Current frame at window position = num_cached.
        if self_attn.qk_rms_norm:
            k_cur = self_attn.k_rms_norm(k_cur_pre_rms)
            q = self_attn.q_rms_norm(q)
        else:
            k_cur = k_cur_pre_rms
        pos_cur = torch.tensor([cur_pos], device=x.device, dtype=torch.long)
        k_cur = self.rope(k_cur, pos_cur)
        q = self.rope(q, pos_cur)
        K_parts.append(k_cur)
        V_parts.append(v_cur_bare)

        # flash_attn_func expects (B, S, H, d) — no transpose dance. Q only
        # contains the current frame; K/V contain past+current; no mask
        # needed because past entries are exactly the valid context.
        k = torch.cat(K_parts, dim=1).contiguous()  # (B, L_total, H, d)
        v = torch.cat(V_parts, dim=1).contiguous()  # (B, L_total, H, d)
        q = q.contiguous()                          # (B, L, H, d)
        out = fa.flash_attn_func(q, k, v)           # (B, L, H, d)
        out = out.reshape(B, L, C)
        out = self_attn.to_out(out)
        # Cache pre-RMS, pre-RoPE K and bare V — frame-position-agnostic.
        return out, k_cur_pre_rms.detach(), v_cur_bare.detach()

    # ------------------------------------------------------------------
    # Block forward helpers
    # ------------------------------------------------------------------
    def _block_forward_causal(
        self,
        block,
        x: torch.Tensor,     # (B, F*L, C)
        mod_per_frame: torch.Tensor,  # (B, F, 6*C) post-adaLN modulation
        cond_per_frame: torch.Tensor,  # (B, F, L_cond, D_cond)
        F_frames: int,
        L: int,
    ) -> torch.Tensor:
        B = x.shape[0]
        C = x.shape[-1]
        # Broadcast per-frame modulation over the L tokens in each frame.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_per_frame.chunk(6, dim=-1)

        def _expand(t):
            # (B, F, C) -> (B, F, 1, C) -> (B, F, L, C) -> (B, F*L, C)
            return t.unsqueeze(2).expand(B, F_frames, L, C).reshape(B, F_frames * L, C)

        shift_msa = _expand(shift_msa); scale_msa = _expand(scale_msa); gate_msa = _expand(gate_msa)
        shift_mlp = _expand(shift_mlp); scale_mlp = _expand(scale_mlp); gate_mlp = _expand(gate_mlp)

        # Self-attn (block-causal via per-frame slicing inside flash_attn).
        h = block.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self._frame_causal_self_attn(block.self_attn, h, F_frames, L)
        h = h * gate_msa
        x = x + h

        # DINOv2 cross-attn per-frame. Reshape (B, F*L, C) -> (B*F, L, C); cond
        # likewise (B, F, L_c, D) -> (B*F, L_c, D). No cross-frame leakage.
        B_f = B * F_frames
        h = block.norm2(x)
        h = h.reshape(B, F_frames, L, C).reshape(B_f, L, C)
        ctx = cond_per_frame.reshape(B_f, cond_per_frame.shape[2], cond_per_frame.shape[3])
        h = block.cross_attn(h, ctx)
        h = h.reshape(B, F_frames, L, C).reshape(B, F_frames * L, C)
        x = x + h

        # FFN
        h = block.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = block.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def _block_forward_kvcache(
        self,
        block,
        x: torch.Tensor,     # (B=1, L, C)
        mod: torch.Tensor,   # (B=1, 6*C)
        cond: torch.Tensor,  # (B=1, L_cond, D_cond)
        cached_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)

        h = block.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, new_k, new_v = self._kvcache_self_attn(block.self_attn, h, cached_list)
        h = h * gate_msa.unsqueeze(1)
        x = x + h

        h = block.norm2(x)
        h = block.cross_attn(h, cond)
        x = x + h

        h = block.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = block.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x, new_k, new_v

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _per_frame_modulation(self, t_per_frame: torch.Tensor) -> torch.Tensor:
        """Returns (B, F, 6*C) post-AdaLN modulation vector for each frame."""
        B, Fr = t_per_frame.shape
        base = self.base_model
        t_flat = t_per_frame.reshape(-1) * 1000
        t_emb = base.t_embedder(t_flat)           # (B*F, C)
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)  # (B*F, 6*C)
        else:
            # Per-block modulation applied inside _block_forward_causal via
            # block.adaLN_modulation. Here we just pass the raw t_emb; the block
            # chunks after projecting itself. Return (B, F, C).
            return t_emb.reshape(B, Fr, -1).to(self.dtype)
        return t_emb.reshape(B, Fr, -1).to(self.dtype)

    # ------------------------------------------------------------------
    # Per-frame input / output paths
    # ------------------------------------------------------------------
    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F, in_ch, R, R, R) -> (B, F, L, C) via shared frozen input path."""
        B, Fr, in_ch, R, _, _ = x.shape
        base = self.base_model
        x_flat = x.reshape(B * Fr, in_ch, R, R, R)
        h = patchify(x_flat, base.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # (B*F, L, in_ch*p^3)
        h = base.input_layer(h)                                     # (B*F, L, C)
        h = h + base.pos_emb[None]
        L = h.shape[1]
        C = h.shape[-1]
        return h.reshape(B, Fr, L, C).to(self.dtype)

    def _decode_frames(self, h: torch.Tensor, out_dtype: torch.dtype, R: int) -> torch.Tensor:
        """(B, F, L, C) -> (B, F, out_ch, R, R, R) via frozen out_layer."""
        B, Fr, L, C = h.shape
        base = self.base_model
        h_flat = h.reshape(B * Fr, L, C).to(out_dtype)
        h_flat = F.layer_norm(h_flat, h_flat.shape[-1:])
        h_flat = base.out_layer(h_flat)  # (B*F, L, out_ch*p^3)
        Rp = R // base.patch_size
        h_flat = h_flat.permute(0, 2, 1).view(h_flat.shape[0], h_flat.shape[2], Rp, Rp, Rp)
        h_flat = unpatchify(h_flat, base.patch_size).contiguous()
        return h_flat.reshape(B, Fr, h_flat.shape[1], R, R, R)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
        kv_caches: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch: 6D input -> multi-frame causal training; 5D -> inference
        single-frame with optional KV cache (called by the sampler).

        Training args:
            x: (B, F, in_ch, R, R, R) noisy latents per frame.
            t: (B, F) noise levels in [0, 1] OR pre-scaled *1000; auto-detected.
            cond: (B, F, L_cond, D_cond) per-frame DINOv2 features.
            frame_indices: (B, F) long tensor of absolute frame positions.

        Inference args:
            x: (1, in_ch, R, R, R)
            t: (1,) pre-scaled *1000 (as provided by the sampler).
            cond: (1, L_cond, D_cond).
            frame_idx: int temporal position of this frame.
            kv_caches: list (oldest-first) of per-frame caches; each cache is a
                list of (k, v) per block.

        Returns:
            Training: pred_v (B, F, out_ch, R, R, R).
            Inference: pred_v (1, out_ch, R, R, R).
        """
        if x.dim() == 5:
            assert frame_idx is not None, (
                "forward() with 5D x requires frame_idx; pass via kwargs from sampler."
            )
            return self.forward_with_kvcache(
                x=x, t=t, cond=cond, frame_idx=frame_idx, kv_caches=kv_caches,
            )
        assert x.dim() == 6, f"Expected (B, F, C, R, R, R); got {x.shape}"
        B, Fr, in_ch, R, _, _ = x.shape
        device = x.device
        base = self.base_model

        # t is assumed to be the raw [0, 1] noise level (the sampler / trainer
        # is responsible for the *1000 scaling when calling t_embedder). Here
        # the trainer passes already-scaled t*1000 into the single-frame path
        # of the base model, but for diff-forcing we keep t in [0,1] across
        # the API and multiply internally.
        if t.dim() == 1:
            t = t.view(B, Fr)
        assert t.shape == (B, Fr), f"t shape {t.shape} != (B, F)={(B, Fr)}"
        # Accept two conventions: [0,1] fractional, or pre-scaled *1000.
        # Anything <= 1.001 is treated as fractional.
        if t.max() <= 1.001:
            t_frac = t
        else:
            t_frac = t / 1000.0

        # Input path (per-frame, frozen). Per-frame noise-level info flows
        # only through adaLN (`_per_frame_modulation`); frame position is
        # carried by RoPE inside self-attention.
        h = self._encode_frames(x)                                # (B, F, L, C)
        L = h.shape[2]
        C = h.shape[3]

        # AdaLN modulation input (per-frame). For share_mod=True, apply the
        # shared projection now (-> 6C); for share_mod=False, pass the C-dim
        # t_emb and let each block project it (inside _block_forward_causal).
        mod_raw = self._per_frame_modulation(t_frac)  # (B, F, 6C) if share_mod else (B, F, C)

        # Flatten to (B, F*L, C) for transformer body
        h_flat = h.reshape(B, Fr * L, C)

        cond = cond.to(self.dtype)
        mod_raw = mod_raw.to(self.dtype)

        for block in base.blocks:
            # Produce per-frame (B, F, 6C) modulation for this block.
            if base.share_mod:
                mod_per_frame = mod_raw  # (B, F, 6C)
            else:
                B_, F_, Cm = mod_raw.shape
                mod_per_frame = block.adaLN_modulation(mod_raw.reshape(B_ * F_, Cm)).reshape(B_, F_, -1)
            if getattr(block, 'use_checkpoint', False) and self.training:
                h_flat = torch.utils.checkpoint.checkpoint(
                    self._block_forward_causal,
                    block, h_flat, mod_per_frame, cond, Fr, L,
                    use_reentrant=False,
                )
            else:
                h_flat = self._block_forward_causal(
                    block, h_flat, mod_per_frame, cond, Fr, L,
                )

        # Reshape back and decode per-frame
        h = h_flat.reshape(B, Fr, L, C)
        pred = self._decode_frames(h, x.dtype, R)  # (B, F, out_ch, R, R, R)
        return pred

    # ------------------------------------------------------------------
    # Inference: single-frame forward with cached K/V from prev frames
    # ------------------------------------------------------------------
    def forward_with_kvcache(
        self,
        x: torch.Tensor,              # (1, in_ch, R, R, R)
        t: torch.Tensor,              # (1,) pre-scaled *1000 (from sampler)
        cond: torch.Tensor,           # (1, L_cond, D_cond)
        frame_idx: int,
        kv_caches: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Denoise ONE frame given KV cache from previously-denoised frames.

        Args:
            frame_idx: absolute frame index; kept for API compat / logging only.
                Window position (what drives RoPE rotation on Q/K) is derived
                at attention time from `len(kv_caches)`.
            kv_caches: list (oldest-first) of per-frame caches; each cache is a
                list of (k_pre_rms, v_bare) per block. If None / empty, no prev
                context. Each cached tensor is frame-position-agnostic — the
                window-position RoPE rotation is applied at attention time.
        """
        assert x.dim() == 5 and x.shape[0] == 1, f"Expected (1, in_ch, R, R, R); got {x.shape}"
        base = self.base_model
        device = x.device
        # t from the sampler is pre-scaled *1000. Convert to [0,1] for the
        # adaLN_modulation input.
        t_scalar = float(t.reshape(-1)[0].item()) / 1000.0

        # Input path (single frame). NOTE: no frame-position addition at input;
        # RoPE is injected inside _kvcache_self_attn at attention time, keyed
        # off the window position (len(kv_caches) for the current frame).
        h = patchify(x, base.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = base.input_layer(h)
        h = h + base.pos_emb[None]
        h = h.to(self.dtype)

        t_emb = base.t_embedder(torch.tensor([t_scalar * 1000.0], device=device))
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = t_emb.to(self.dtype)
        cond = cond.to(self.dtype)

        for blk_idx, block in enumerate(base.blocks):
            cached_list = (
                [(c[blk_idx][0], c[blk_idx][1]) for c in kv_caches] if kv_caches else []
            )
            if base.share_mod:
                mod = t_emb
            else:
                mod = block.adaLN_modulation(t_emb)
            h, _, _ = self._block_forward_kvcache(block, h, mod, cond, cached_list)

        # Output path
        h = h.to(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = base.out_layer(h)
        Rp = x.shape[-1] // base.patch_size
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], Rp, Rp, Rp)
        h = unpatchify(h, base.patch_size).contiguous()
        return h

    @torch.no_grad()
    def build_kv_cache(
        self,
        x: torch.Tensor,       # (1, in_ch, R, R, R) clean or denoised
        frame_idx: int,
        cond: torch.Tensor,    # (1, L_cond, D_cond)
        t_noise: float = 0.0,
        prev_kv_caches: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run one frame through all blocks and collect per-block (K_pre_rms,
        V_bare) as the cache entries for that frame. The returned tensors are
        frame-position-agnostic: the window-position contribution is injected
        at attention time (see `_kvcache_self_attn`), so these cache entries
        stay valid as the sliding window advances.

        Use after the frame is fully denoised (or on GT for teacher-forcing).

        Args:
            frame_idx: kept for API compat / logging only; not consumed here.
            prev_kv_caches: oldest-first list of per-frame caches built for
                the preceding frames in this window; the new frame's window
                slot is ``len(prev_kv_caches)``.
        """
        base = self.base_model
        device = x.device

        h = patchify(x, base.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = base.input_layer(h)
        h = h + base.pos_emb[None]
        h = h.to(self.dtype)

        t_emb = base.t_embedder(torch.tensor([t_noise * 1000.0], device=device))
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = t_emb.to(self.dtype)
        cond = cond.to(self.dtype)

        kv_new: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk_idx, block in enumerate(base.blocks):
            cached_list = (
                [(c[blk_idx][0], c[blk_idx][1]) for c in prev_kv_caches] if prev_kv_caches else []
            )
            if base.share_mod:
                mod = t_emb
            else:
                mod = block.adaLN_modulation(t_emb)
            h, new_k, new_v = self._block_forward_kvcache(block, h, mod, cond, cached_list)
            kv_new.append((new_k, new_v))
        return kv_new
