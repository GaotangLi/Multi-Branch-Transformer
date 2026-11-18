# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Optional inference-only elementwise kernels; not enabled by the model.

Each operation preserves the current model's FP32 arithmetic and returns BF16.
RoPE uses CUDA libdevice trigonometry, including at large absolute positions.
Packed QKV views are accepted without making an intermediate contiguous copy.
GPU numerical and timing qualification is required before adopting these ops.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = libdevice = None


def rmsnorm_reference(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).to(x.dtype)


def rope_reference(x, positions, invfreq):
    phase = positions.float().unsqueeze(-1) * invfreq.float()
    cosine, sine = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine,
                      second * cosine + first * sine), dim=-1).to(x.dtype)


def gate_reference(attention, gate):
    return (attention.float() * torch.sigmoid(gate.reshape_as(attention).float())).to(attention.dtype)


def _layout(x):
    if x.ndim == 2:
        tokens, width = x.shape
        heads, head_stride = 1, 0
    elif x.ndim == 3:
        tokens, heads, width = x.shape
        head_stride = x.stride(1)
    else:
        raise ValueError("expected [tokens,width] or [tokens,heads,width]")
    if heads < 1 or width < 1:
        raise ValueError("head count and width must be positive")
    return tokens, heads, width, x.stride(0), head_stride, x.stride(-1)


def _cuda_bf16(x, *others):
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError("production ops require CUDA BF16; call explicit references for CPU")
    if any(t.device != x.device for t in others):
        raise ValueError("all operands must be on the same device")
    if triton is None:
        raise RuntimeError("production ops require Triton")


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(X, Weight, Out, HEADS: tl.constexpr, WIDTH: tl.constexpr,
                        TOKEN_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr,
                        COL_STRIDE: tl.constexpr, WEIGHT_STRIDE: tl.constexpr,
                        EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        offset = (row // HEADS) * TOKEN_STRIDE + (row % HEADS) * HEAD_STRIDE + col * COL_STRIDE
        value = tl.load(X + offset, col < WIDTH, other=0).to(tl.float32)
        weight = tl.load(Weight + col * WEIGHT_STRIDE, col < WIDTH, other=0).to(tl.float32)
        mean_square = tl.sum(value * value, axis=0) / WIDTH
        normalized = value * libdevice.rsqrt(mean_square + EPS)
        output = (normalized * weight).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + row * WIDTH + col, output, col < WIDTH)

    @triton.jit
    def _rope_kernel(X, Positions, InvFreq, Out, HEADS: tl.constexpr, WIDTH: tl.constexpr,
                     TOKEN_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr,
                     COL_STRIDE: tl.constexpr, POSITION_STRIDE: tl.constexpr,
                     FREQ_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        half: tl.constexpr = WIDTH // 2
        base = (row // HEADS) * TOKEN_STRIDE + (row % HEADS) * HEAD_STRIDE
        first = tl.load(X + base + col * COL_STRIDE, col < half, other=0).to(tl.float32)
        second = tl.load(X + base + (col + half) * COL_STRIDE, col < half, other=0).to(tl.float32)
        position = tl.load(Positions + (row // HEADS) * POSITION_STRIDE).to(tl.float32)
        frequency = tl.load(InvFreq + col * FREQ_STRIDE, col < half, other=0).to(tl.float32)
        phase = position * frequency
        # tl.cos/tl.sin use fast approximations whose range reduction is not
        # sufficient evidence for million-token positions. Use libdevice.
        cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
        rotated_first = (first * cosine - second * sine).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        rotated_second = (second * cosine + first * sine).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + row * WIDTH + col, rotated_first, col < half)
        tl.store(Out + row * WIDTH + col + half, rotated_second, col < half)

    @triton.jit
    def _gate_kernel(Attention, Gate, Out, NUMEL: tl.constexpr,
                     FEATURES: tl.constexpr, ATTN_WIDTH: tl.constexpr, GATE_WIDTH: tl.constexpr,
                     ATTN_TOKEN_STRIDE: tl.constexpr, ATTN_HEAD_STRIDE: tl.constexpr,
                     ATTN_COL_STRIDE: tl.constexpr, GATE_TOKEN_STRIDE: tl.constexpr,
                     GATE_HEAD_STRIDE: tl.constexpr, GATE_COL_STRIDE: tl.constexpr,
                     BLOCK: tl.constexpr):
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        token, feature = offset // FEATURES, offset % FEATURES
        attn_offset = token * ATTN_TOKEN_STRIDE + (feature // ATTN_WIDTH) * ATTN_HEAD_STRIDE + (feature % ATTN_WIDTH) * ATTN_COL_STRIDE
        gate_offset = token * GATE_TOKEN_STRIDE + (feature // GATE_WIDTH) * GATE_HEAD_STRIDE + (feature % GATE_WIDTH) * GATE_COL_STRIDE
        attention = tl.load(Attention + attn_offset, offset < NUMEL, other=0).to(tl.float32)
        gate = tl.load(Gate + gate_offset, offset < NUMEL, other=0).to(tl.float32)
        sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        output = (attention * sigmoid).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + offset, output, offset < NUMEL)


def fused_rmsnorm(x, weight, eps):
    """Normalize the last axis of 2D/3D CUDA BF16 input in one kernel."""
    tokens, heads, width, token_stride, head_stride, col_stride = _layout(x)
    if weight.ndim != 1 or weight.numel() != width:
        raise ValueError("RMSNorm weight must match the final input dimension")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("RMSNorm weight must be FP32 or BF16")
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be a positive finite scalar")
    _cuda_bf16(x, weight)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens:
        block = triton.next_power_of_2(width)
        _rmsnorm_kernel[(tokens * heads,)](
            x, weight, output, heads, width, token_stride, head_stride, col_stride,
            weight.stride(0), float(eps), block, num_warps=4 if block <= 4096 else 8,
            enable_fp_fusion=False)
    return output


def fused_rope(x, positions, invfreq):
    """Half-rotate [tokens,heads,head_dim] with FP32 libdevice sin/cos."""
    if x.ndim != 3 or x.shape[-1] % 2:
        raise ValueError("RoPE requires [tokens,heads,even head_dim]")
    tokens, heads, width, token_stride, head_stride, col_stride = _layout(x)
    if positions.ndim != 1 or positions.numel() != tokens:
        raise ValueError("one position is required per token")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must remain integer token positions")
    if invfreq.ndim != 1 or invfreq.numel() != width // 2 or invfreq.dtype != torch.float32:
        raise ValueError("RoPE inverse frequencies must be FP32 with head_dim/2 entries")
    _cuda_bf16(x, positions, invfreq)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens:
        _rope_kernel[(tokens * heads,)](
            x, positions, invfreq, output, heads, width, token_stride, head_stride, col_stride,
            positions.stride(0), invfreq.stride(0), triton.next_power_of_2(width // 2),
            num_warps=4, enable_fp_fusion=False)
    return output


def fused_gate(attention, gate):
    """FP32 sigmoid gating, accepting the native strided [T,KV,2*D] gate."""
    tokens, heads, width, ts, hs, cs = _layout(attention)
    gate_tokens, gate_heads, gate_width, gts, ghs, gcs = _layout(gate)
    if tokens != gate_tokens or heads * width != gate_heads * gate_width:
        raise ValueError("attention and gate must have the same tokens and features")
    if gate.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("attention gate must be FP32 or BF16")
    _cuda_bf16(attention, gate)
    output = torch.empty(attention.shape, dtype=attention.dtype, device=attention.device)
    if tokens:
        _gate_kernel[(triton.cdiv(attention.numel(), 1024),)](
            attention, gate, output, attention.numel(), heads * width, width, gate_width,
            ts, hs, cs, gts, ghs, gcs, 1024, enable_fp_fusion=False)
    return output


if triton is not None:

    @triton.jit
    def _rmsnorm_stacked_kernel(X, Weight, Out, Cat, scale, TOKENS: tl.constexpr, WIDTH: tl.constexpr, N: tl.constexpr,
                                WEIGHT_STRIDE: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr, CAT: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        value = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        weight = tl.load(Weight + (row // TOKENS) * WEIGHT_STRIDE + col, col < WIDTH, other=0).to(tl.float32)
        normalized = value * libdevice.rsqrt(tl.sum(value * value, axis=0) / WIDTH + EPS)
        output = (normalized * weight).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + row * WIDTH + col, output, col < WIDTH)
        if CAT:  # also torch.cat(out.unbind(0), -1) * scale: branch b's row t at Cat[t, b*WIDTH:]
            scaled = (output.to(tl.float32) * scale).to(Cat.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(Cat + (row % TOKENS) * (N * WIDTH) + (row // TOKENS) * WIDTH + col, scaled, col < WIDTH)

    @triton.jit
    def _gate_unfold_kernel(Attention, Gate, Out, NUMEL, TOKENS, FEATURES: tl.constexpr, GROUPS: tl.constexpr,
                            N: tl.constexpr, D: tl.constexpr, GATE_TOKEN_STRIDE: tl.constexpr,
                            GATE_HEAD_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
        # _gate_kernel arithmetic; attention is read from the folded layout written by norm_rope_heads(branches=N).
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        token, feature = offset // FEATURES, offset % FEATURES
        branch, t = token // TOKENS, token % TOKENS
        head, col = feature // D, feature % D
        attn_offset = ((t * GROUPS + head // 2) * N * 2 + branch * 2 + head % 2) * D + col
        gate_offset = token * GATE_TOKEN_STRIDE + (feature // (2 * D)) * GATE_HEAD_STRIDE + feature % (2 * D)
        attention = tl.load(Attention + attn_offset, offset < NUMEL, other=0).to(tl.float32)
        gate = tl.load(Gate + gate_offset, offset < NUMEL, other=0).to(tl.float32)
        sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(Out + offset, (attention * sigmoid).to(Out.dtype.element_ty, fp_downcast_rounding="rtne"), offset < NUMEL)


def fused_rmsnorm_stacked(x, weight, eps, cat_scale=None):
    """fused_rmsnorm arithmetic for stacked branches: x [n, T, width] contiguous, weight [n, width].
    With cat_scale also returns torch.cat(out.unbind(0), -1) * cat_scale ([T, n*width]) from the same pass."""
    n, tokens, width = x.shape
    output = torch.empty_like(x)
    cat = output if cat_scale is None else torch.empty((tokens, n * width), dtype=x.dtype, device=x.device)
    if tokens:
        block = triton.next_power_of_2(width)
        _rmsnorm_stacked_kernel[(n * tokens,)](x, weight, output, cat, float(cat_scale or 0), tokens, width, n,
                                               weight.stride(0), float(eps), block, cat_scale is not None,
                                               num_warps=4 if block <= 4096 else 8, enable_fp_fusion=False)
    return output if cat_scale is None else (output, cat)


def fused_gate_unfold(attention, gate, branches):
    """fused_gate for stacked branches whose attention output is folded ([T, KV*n*2, D], see norm_rope_heads);
    gate is the strided [n*T, KV, 2D] view of the packed rows. Returns [n*T, 2KV, D]."""
    tokens, heads, d = attention.shape
    output = torch.empty((branches * tokens, heads // branches, d), dtype=attention.dtype, device=attention.device)
    if output.numel():
        _gate_unfold_kernel[(triton.cdiv(output.numel(), 1024),)](
            attention, gate, output, output.numel(), tokens, heads // branches * d, heads // (2 * branches), branches, d,
            gate.stride(0), gate.stride(1), 1024, enable_fp_fusion=False)
    return output
