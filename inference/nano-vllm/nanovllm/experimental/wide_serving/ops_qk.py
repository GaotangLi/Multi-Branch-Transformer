# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Packed Q/K RMSNorm + RoPE Triton kernels.

The intermediate BF16 rounding between norm and RoPE is intentional. This
kernel preserves grouped [Q(2D), gate(2D), K(D), V(D)] checkpoint packing and
uses FP32 libdevice trig, not an approximate-position shortcut. The standalone
candidate was qualified on the installed stack; a new stack still needs GPU
and full-model qualification. No CPU/unsupported-shape fallback is implicit.
"""
import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = libdevice = None


def _check(packed, qweight, kweight, positions, invfreq, eps):
    if (packed.ndim != 3 or packed.shape[-1] != 384 or not packed.is_contiguous()
            or packed.dtype != torch.bfloat16):
        raise ValueError('Requires contiguous BF16 [tokens,KV,384]')
    tokens, kv, _ = packed.shape
    if (kv < 1 or qweight.shape != (64,) or kweight.shape != (64,) or positions.shape != (tokens,)
            or qweight.dtype not in (torch.bfloat16, torch.float32)
            or kweight.dtype not in (torch.bfloat16, torch.float32)
            or invfreq.shape != (32,) or invfreq.dtype != torch.float32
            or positions.dtype not in (torch.int32, torch.int64)
            or any(t.device != packed.device or not t.is_contiguous()
                   for t in (qweight, kweight, positions, invfreq))):
        raise ValueError('Invalid Q/K weight, position or RoPE metadata')
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Positive finite normalization epsilon required')


if triton is not None:

    @triton.jit
    def _qk(Packed, Qweight, Kweight, Positions, Invfreq, Qout, Kout,
            KV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        token, head = row // (3*KV), row % (3*KV)
        is_q = head < 2*KV
        group = tl.where(is_q, head // 2, head - 2*KV)
        in_head = tl.where(is_q, (head % 2)*D, 4*D)
        col = tl.arange(0, D)
        value = tl.load(Packed + (token*KV + group)*6*D + in_head + col).to(tl.float32)
        weight = tl.load(tl.where(is_q, Qweight, Kweight) + col).to(tl.float32)
        mean_square = tl.sum(value*value, 0) / D
        normalized = value * libdevice.rsqrt(mean_square + EPS)
        normalized = (normalized*weight).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        other = tl.gather(normalized, (col + D//2) % D, 0)
        position = tl.load(Positions + token).to(tl.float32)
        frequency = tl.load(Invfreq + col % (D//2))
        phase = position*frequency
        cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
        rotated = tl.where(col < D//2, normalized*cosine-other*sine,
                           normalized*cosine+other*sine).to(tl.bfloat16, fp_downcast_rounding='rtne')
        tl.store(Qout + (token*2*KV + head)*D + col, rotated, is_q)
        tl.store(Kout + (token*KV + head-2*KV)*D + col, rotated, ~is_q)

    @triton.jit
    def _qk_rows(Packed, Qweight, Kweight, Positions, Invfreq, Qout, Kout,
                 KV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
        # The _qk body, looped over the token's 3*KV heads in one program.
        token = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, D)
        position = tl.load(Positions + token).to(tl.float32)
        for head in range(3*KV):
            is_q = head < 2*KV
            group = tl.where(is_q, head // 2, head - 2*KV)
            in_head = tl.where(is_q, (head % 2)*D, 4*D)
            value = tl.load(Packed + (token*KV + group)*6*D + in_head + col).to(tl.float32)
            weight = tl.load(tl.where(is_q, Qweight, Kweight) + col).to(tl.float32)
            mean_square = tl.sum(value*value, 0) / D
            normalized = value * libdevice.rsqrt(mean_square + EPS)
            normalized = (normalized*weight).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
            other = tl.gather(normalized, (col + D//2) % D, 0)
            frequency = tl.load(Invfreq + col % (D//2))
            phase = position*frequency
            cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
            rotated = tl.where(col < D//2, normalized*cosine-other*sine,
                               normalized*cosine+other*sine).to(tl.bfloat16, fp_downcast_rounding='rtne')
            tl.store(Qout + (token*2*KV + head)*D + col, rotated, mask=(col < D) & is_q)
            tl.store(Kout + (token*KV + head-2*KV)*D + col, rotated, mask=(col < D) & (head >= 2*KV))

    @triton.jit
    def _norm_rope_heads(X, Weight, Positions, Invfreq, Out, tokens_per_batch,
                         GROUPS: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, BRANCHES: tl.constexpr,
                         TOKEN_STRIDE: tl.constexpr, GROUP_STRIDE: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
        # The _qk arithmetic for head h of group g at X[t, g, (h % HEADS_PER_GROUP)*D : +D].
        row = tl.program_id(0).to(tl.int64)
        heads = GROUPS*HEADS_PER_GROUP
        token, head = row // heads, row % heads
        group, within = head // HEADS_PER_GROUP, head % HEADS_PER_GROUP
        col = tl.arange(0, D)
        value = tl.load(X + token*TOKEN_STRIDE + group*GROUP_STRIDE + within*D + col).to(tl.float32)
        weight = tl.load(Weight + (token // tokens_per_batch)*D + col).to(tl.float32)
        mean_square = tl.sum(value*value, 0) / D
        normalized = value * libdevice.rsqrt(mean_square + EPS)
        normalized = (normalized*weight).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        other = tl.gather(normalized, (col + D//2) % D, 0)
        position = tl.load(Positions + token).to(tl.float32)
        frequency = tl.load(Invfreq + col % (D//2))
        phase = position*frequency
        cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
        rotated = tl.where(col < D//2, normalized*cosine-other*sine,
                           normalized*cosine+other*sine).to(tl.bfloat16, fp_downcast_rounding='rtne')
        # Branch b = token // tokens_per_batch writes slot b of its group's BRANCHES*HEADS_PER_GROUP heads.
        branch, t = token // tokens_per_batch, token % tokens_per_batch
        tl.store(Out + ((t*GROUPS + group)*BRANCHES*HEADS_PER_GROUP + branch*HEADS_PER_GROUP + within)*D + col, rotated)

    @triton.jit
    def _norm_rope_store_kv(KV, Weight, Positions, Invfreq, Slots, KC, VC, GROUPS: tl.constexpr,
                            TOKEN_STRIDE: tl.constexpr, GROUP_STRIDE: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
        # K: the _norm_rope_heads arithmetic; V: copied; both stored to cache row `slot` as store_kv does.
        row = tl.program_id(0).to(tl.int64)
        token, group = row // GROUPS, row % GROUPS
        col = tl.arange(0, D)
        base = KV + token*TOKEN_STRIDE + group*GROUP_STRIDE
        value = tl.load(base + col).to(tl.float32)
        weight = tl.load(Weight + col).to(tl.float32)
        mean_square = tl.sum(value*value, 0) / D
        normalized = value * libdevice.rsqrt(mean_square + EPS)
        normalized = (normalized*weight).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        other = tl.gather(normalized, (col + D//2) % D, 0)
        position = tl.load(Positions + token).to(tl.float32)
        frequency = tl.load(Invfreq + col % (D//2))
        phase = position*frequency
        cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
        rotated = tl.where(col < D//2, normalized*cosine-other*sine,
                           normalized*cosine+other*sine).to(tl.bfloat16, fp_downcast_rounding='rtne')
        slot = tl.load(Slots + token).to(tl.int64)
        mask = (col < D) & (slot >= 0)
        tl.store(KC + slot*(GROUPS*D) + group*D + col, rotated, mask=mask)
        tl.store(VC + slot*(GROUPS*D) + group*D + col, tl.load(base + D + col), mask=mask)


def qk_fused(packed, qweight, kweight, positions, invfreq, eps=1e-5):
    _check(packed, qweight, kweight, positions, invfreq, eps)
    if not packed.is_cuda or triton is None:
        raise RuntimeError('packed Q/K requires CUDA and Triton')
    tokens, kv, _ = packed.shape
    query = torch.empty((tokens, 2*kv, 64), dtype=packed.dtype, device=packed.device)
    key = torch.empty((tokens, kv, 64), dtype=packed.dtype, device=packed.device)
    if tokens:
        kernel, grid = (_qk_rows, tokens) if tokens >= 2048 else (_qk, tokens*3*kv)  # long prefill: one program per token
        kernel[(grid,)](packed, qweight, kweight, positions, invfreq, query, key, kv, 64, float(eps),
                        num_warps=4, enable_fp_fusion=False)
    return query, key


def qk_reference(packed, qweight, kweight, positions, invfreq, eps=1e-5):
    """Explicit CPU/CUDA Torch mathematical reference, not a production fallback.

    FP32 Torch reductions/trig need not be bitwise the separate Triton kernels;
    GPU exact-identity tests compare directly against those existing kernels.
    """
    _check(packed, qweight, kweight, positions, invfreq, eps)
    from .ops import rmsnorm_reference, rope_reference
    tokens, kv, _ = packed.shape
    q, _, k, _ = packed.split((128, 128, 64, 64), -1)
    q = rmsnorm_reference(q.reshape(tokens, 2*kv, 64), qweight, eps)
    k = rmsnorm_reference(k, kweight, eps)
    return rope_reference(q, positions, invfreq), rope_reference(k, positions, invfreq)


def norm_rope_heads(parent, weight, positions, invfreq, eps, *, heads_per_group, branches=1):
    """RMSNorm + RoPE of the first heads_per_group 64-wide heads of each group of a contiguous BF16
    [T, groups, width] parent (Q of a Q/gate row: 2; K of a bank's K/V row: 1). With branches=n the rows are
    n stacked branches (weight [n, 64]) and the output is folded: [T/n, groups*n*heads_per_group, 64], the
    n branches' heads of each group adjacent, i.e. one GQA group per KV head."""
    tokens, groups, width = parent.shape
    out = torch.empty((tokens // branches, groups*branches*heads_per_group, 64), dtype=parent.dtype, device=parent.device)
    if tokens:
        _norm_rope_heads[(tokens*groups*heads_per_group,)](
            parent, weight, positions, invfreq, out, tokens // branches, groups, heads_per_group, branches,
            groups*width, width, 64, float(eps), num_warps=4, enable_fp_fusion=False)
    return out


def norm_rope_store_kv(kv, weight, positions, invfreq, eps, kc, vc, slots):
    """One kernel for a bank's [K | V] rows kv [T, groups, 128]: K RMSNorm + RoPE, then K and V into cache rows `slots`."""
    tokens, groups, width = kv.shape
    if tokens:
        _norm_rope_store_kv[(tokens*groups,)](kv, weight, positions, invfreq, slots, kc, vc, groups, groups*width,
                                              width, 64, float(eps), num_warps=4, enable_fp_fusion=False)
