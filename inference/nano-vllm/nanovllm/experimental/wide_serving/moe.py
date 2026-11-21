# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Graph-compatible BF16 experts for the Wide/Fast-Slow inference experiment.

Weights follow the training layout: w1[E, 2F, D] has gate then up rows,
and w2[E, D, F]. Routing is external. In particular, probabilities multiply
the activated intermediate BEFORE its BF16 cast and FC2, as in the native
torch.compile-fused weighted SwiGLU. FC1 and FC2 outputs remain BF16.

The grouped backend uses PyTorch's native grouped GEMM (no Python expert
loop); the decode backend issues parallel Triton GEMVs for routed experts; the
triton backend (9-256 tokens) reads each expert weight tile once for all of its
rows and fuses weighted SwiGLU into the first GEMM.
``reference`` is a correctness oracle, never a throughput backend. Backend
selection depends only on tensor shapes, so replay may change routing IDs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


def _validate(x, expert_indices, router_weights, w1, w2):
    if x.ndim != 2 or expert_indices.ndim != 2:
        raise ValueError("x and expert_indices must have shapes [tokens,D] and [tokens,K]")
    if router_weights.shape != expert_indices.shape or expert_indices.shape[0] != x.shape[0]:
        raise ValueError("router weights and indices must have the same [tokens,K] shape")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[1] != 2 * w2.shape[2]:
        raise ValueError("expected w1[E,2F,D] and w2[E,D,F]")
    if w1.shape[0] != w2.shape[0] or w1.shape[2] != x.shape[1] or w2.shape[1] != x.shape[1]:
        raise ValueError("expert counts and latent dimensions must match")
    if expert_indices.shape[1] < 1 or w1.shape[0] < 1:
        raise ValueError("at least one selected expert and one expert are required")
    if expert_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert indices must be int32 or int64")
    if not router_weights.is_floating_point():
        raise TypeError("router weights must be floating point")
    if any(t.device != x.device for t in (expert_indices, router_weights, w1, w2)):
        raise ValueError("all operands must be on the same device")
    if x.dtype != w1.dtype or x.dtype != w2.dtype:
        raise TypeError("x, w1, and w2 must have the same dtype")
    if not all(t.is_contiguous() for t in (x, expert_indices, router_weights, w1, w2)):
        raise ValueError("all operands must be contiguous; pack expert shards once at load time")


def weighted_swiglu(projected: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """FP32 fused activation/weighting, followed by the native BF16 FC2 input."""
    gate, up = projected.float().chunk(2, dim=-1)
    activated = F.silu(gate) * up
    return (activated * probabilities.unsqueeze(-1)).to(projected.dtype)


def moe_reference(x, expert_indices, router_weights, w1, w2):
    """Small correctness reference; intentionally not suitable for benchmarks."""
    _validate(x, expert_indices, router_weights, w1, w2)
    tokens, top_k = expert_indices.shape
    # Expert loops are confined to the oracle; neither production path uses one.
    routed = torch.empty((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    for expert in range(w1.shape[0]):
        token_ids, choices = torch.where(expert_indices == expert)
        projected = F.linear(x[token_ids], w1[expert])
        hidden = weighted_swiglu(projected, router_weights[token_ids, choices])
        routed[token_ids, choices] = F.linear(hidden, w2[expert])
    return routed.float().sum(dim=1).to(x.dtype)


if triton is not None:

    @triton.jit
    def _route_rank(Ids, Rank, Counts, N, BLOCK: tl.constexpr):
        # Rank of each route within its expert, in (token, choice) order, i.e. a stable sort.
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        base = tl.full((), 0, tl.int32)
        for start in range(0, N, BLOCK):
            ix = start + offs
            ids = tl.load(Ids + ix, ix < N, other=-1)
            hit = (ids == expert).to(tl.int32)
            tl.store(Rank + ix, base + tl.cumsum(hit, 0) - 1, (ix < N) & (hit > 0))
            base += tl.sum(hit, 0)
        tl.store(Counts + expert, base)

    @triton.jit
    def _route_scatter(X, Ids, Rank, Counts, Probs, Selected, SortedProbs, Inverse, Offsets,
                       D: tl.constexpr, K: tl.constexpr, E: tl.constexpr, BD: tl.constexpr, EP: tl.constexpr):
        # EP = next power of two >= E (e.g. 3 stacked branches x 64 = 192 experts); padded lanes count 0.
        route = tl.program_id(0).to(tl.int64)
        dblk = tl.program_id(1)
        e = tl.arange(0, EP)
        counts = tl.load(Counts + e, e < E, other=0)
        inclusive = tl.cumsum(counts, 0)
        expert = tl.load(Ids + route).to(tl.int32)
        start = tl.sum(tl.where(e == expert, inclusive - counts, 0), 0)
        pos = (start + tl.load(Rank + route)).to(tl.int64)
        d = dblk * BD + tl.arange(0, BD)
        tl.store(Selected + pos * D + d, tl.load(X + (route // K) * D + d, d < D), d < D)
        if dblk == 0:
            tl.store(Inverse + route, pos)
            tl.store(SortedProbs + pos, tl.load(Probs + route))
            if route == 0:
                tl.store(Offsets + e, inclusive, e < E)

    @triton.jit
    def _weighted_swiglu_kernel(Y, P, H, FWIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr, B: tl.constexpr):
        # Widen BEFORE the stride product: B256*S2048*K8*2F exceeds signed32.
        row = tl.program_id(0).to(tl.int64)
        off = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
        gate = tl.load(Y + row * ROW_STRIDE + off, off < FWIDTH, other=0).to(tl.float32)
        up = tl.load(Y + row * ROW_STRIDE + FWIDTH + off, off < FWIDTH, other=0).to(tl.float32)
        # Native weighted_swiglu is torch.compile-fused: one final BF16 cast.
        activated = (gate * tl.sigmoid(gate)) * up
        probability = tl.load(P + row).to(tl.float32)
        hidden = (activated * probability).to(H.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(H + row * FWIDTH + off, hidden, off < FWIDTH)

    @triton.jit
    def _gather_reduce(Routed, Inverse, Out, D: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr,
                       BK: tl.constexpr, BD: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BD + tl.arange(0, BD)
        choices = tl.arange(0, BK)
        rows = tl.load(Inverse + token * K + choices, choices < K, other=0).to(tl.int64)
        values = tl.load(Routed + rows[:, None] * STRIDE + d[None, :],
                         (choices[:, None] < K) & (d[None, :] < D), other=0).to(tl.float32)
        summed = tl.sum(values, axis=0).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + token * D + d, summed, d < D)

    @triton.jit
    def _gather_reduce_rows(Routed, Inverse, Out, D: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr,
                            BK: tl.constexpr, BD: tl.constexpr):
        # _gather_reduce with the D blocks looped in-program (long prefill chunks: fewer programs).
        token = tl.program_id(0).to(tl.int64)
        choices = tl.arange(0, BK)
        rows = tl.load(Inverse + token * K + choices, choices < K, other=0).to(tl.int64)
        for start in range(0, D, BD):
            d = start + tl.arange(0, BD)
            values = tl.load(Routed + rows[:, None] * STRIDE + d[None, :],
                             (choices[:, None] < K) & (d[None, :] < D), other=0).to(tl.float32)
            summed = tl.sum(values, axis=0).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(Out + token * D + d, summed, d < D)

    @triton.jit
    def _moe_up_swiglu(Sel, W1, Probs, Offsets, Hidden, D: tl.constexpr, F: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # One program per (expert, N tile); GEMM outputs are rounded to BF16 before SwiGLU, as in _grouped_moe.
        e = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        end = tl.load(Offsets + e)
        start = tl.load(Offsets + e - 1, mask=e > 0, other=0)
        nmask = n < F
        wbase = W1 + e.to(tl.int64) * (2 * F * D)
        for m0 in range(start, end, BM):
            rows = m0 + tl.arange(0, BM)
            rmask = rows < end
            acc_g = tl.zeros((BM, BN), dtype=tl.float32)
            acc_u = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, D, BK):
                k = k0 + tl.arange(0, BK)
                kmask = k < D
                a = tl.load(Sel + rows[:, None].to(tl.int64) * D + k[None, :], rmask[:, None] & kmask[None, :], other=0)
                wg = tl.load(wbase + n[:, None] * D + k[None, :], nmask[:, None] & kmask[None, :], other=0)
                wu = tl.load(wbase + (F + n[:, None]) * D + k[None, :], nmask[:, None] & kmask[None, :], other=0)
                acc_g += tl.dot(a, tl.trans(wg))
                acc_u += tl.dot(a, tl.trans(wu))
            gate = acc_g.to(tl.bfloat16).to(tl.float32)
            up = acc_u.to(tl.bfloat16).to(tl.float32)
            prob = tl.load(Probs + rows, rmask, other=0).to(tl.float32)
            h = ((gate * tl.sigmoid(gate)) * up * prob[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne")
            tl.store(Hidden + rows[:, None].to(tl.int64) * F + n[None, :], h, rmask[:, None] & nmask[None, :])

    @triton.jit
    def _moe_down(Hidden, W2, Offsets, Routed, F: tl.constexpr, DOUT: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        e = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        end = tl.load(Offsets + e)
        start = tl.load(Offsets + e - 1, mask=e > 0, other=0)
        nmask = n < DOUT
        wbase = W2 + e.to(tl.int64) * (DOUT * F)
        for m0 in range(start, end, BM):
            rows = m0 + tl.arange(0, BM)
            rmask = rows < end
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, F, BK):
                k = k0 + tl.arange(0, BK)
                kmask = k < F
                a = tl.load(Hidden + rows[:, None].to(tl.int64) * F + k[None, :], rmask[:, None] & kmask[None, :], other=0)
                w = tl.load(wbase + n[:, None] * F + k[None, :], nmask[:, None] & kmask[None, :], other=0)
                acc += tl.dot(a, tl.trans(w))
            tl.store(Routed + rows[:, None].to(tl.int64) * DOUT + n[None, :],
                     acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), rmask[:, None] & nmask[None, :])

    @triton.jit
    def _decode_up(X, Ids, Probs, W1, H, D: tl.constexpr, FWIDTH: tl.constexpr,
                   K: tl.constexpr, BD: tl.constexpr, BF: tl.constexpr):
        route = tl.program_id(0).to(tl.int64)
        f = tl.program_id(1).to(tl.int64) * BF + tl.arange(0, BF)
        d = tl.arange(0, BD)
        expert = tl.load(Ids + route).to(tl.int64)
        x = tl.load(X + (route // K) * D + d, d < D, other=0).to(tl.float32)
        dtype = H.dtype.element_ty
        gate = tl.load(W1 + expert * (2 * FWIDTH * D) + f[:, None] * D + d[None, :],
                       (f[:, None] < FWIDTH) & (d[None, :] < D), other=0).to(tl.float32)
        gate = tl.sum(gate * x[None, :], 1).to(dtype, fp_downcast_rounding="rtne").to(tl.float32)
        up = tl.load(W1 + expert * (2 * FWIDTH * D) + (FWIDTH + f[:, None]) * D + d[None, :],
                     (f[:, None] < FWIDTH) & (d[None, :] < D), other=0).to(tl.float32)
        up = tl.sum(up * x[None, :], 1).to(dtype, fp_downcast_rounding="rtne").to(tl.float32)
        activated = (gate * tl.sigmoid(gate)) * up
        probability = tl.load(Probs + route).to(tl.float32)
        hidden = (activated * probability).to(dtype, fp_downcast_rounding="rtne")
        tl.store(H + route * FWIDTH + f, hidden, f < FWIDTH)

    @triton.jit
    def _decode_down(H, Ids, W2, Routed, D: tl.constexpr, FWIDTH: tl.constexpr,
                     BF: tl.constexpr, BD: tl.constexpr):
        route = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BD + tl.arange(0, BD)
        f = tl.arange(0, BF)
        expert = tl.load(Ids + route).to(tl.int64)
        h = tl.load(H + route * FWIDTH + f, f < FWIDTH, other=0).to(tl.float32)
        w = tl.load(W2 + expert * D * FWIDTH + d[:, None] * FWIDTH + f[None, :],
                    (d[:, None] < D) & (f[None, :] < FWIDTH), other=0).to(tl.float32)
        result = tl.sum(w * h[None, :], 1).to(Routed.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Routed + route * D + d, result, d < D)

    @triton.jit
    def _decode_reduce(Routed, Out, D: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BD + tl.arange(0, BD)
        k = tl.arange(0, BK)
        values = tl.load(Routed + (token * K + k[:, None]) * D + d[None, :],
                         (k[:, None] < K) & (d[None, :] < D), other=0).to(tl.float32)
        summed = tl.sum(values, 0).to(Out.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Out + token * D + d, summed, d < D)


def _route_plan(x, expert_indices, router_weights, experts):
    """Rows grouped by expert (stable), their probabilities, per-expert end offsets and the inverse map."""
    flat_ids = expert_indices.view(-1)
    n, width = flat_ids.numel(), x.shape[1]
    rank = torch.empty(n, device=x.device, dtype=torch.int32)
    counts = torch.empty(experts, device=x.device, dtype=torch.int32)
    _route_rank[(experts,)](flat_ids, rank, counts, n, 1024, num_warps=4)
    selected = torch.empty((n, width), device=x.device, dtype=x.dtype)
    probabilities = torch.empty(n, device=x.device, dtype=router_weights.dtype)
    inverse = torch.empty(n, device=x.device, dtype=torch.int64)
    offsets = torch.empty(experts, device=x.device, dtype=torch.int32)
    _route_scatter[(n, triton.cdiv(width, 256))](
        x, flat_ids, rank, counts, router_weights.view(-1), selected, probabilities, inverse, offsets,
        width, expert_indices.shape[1], experts, 256, triton.next_power_of_2(experts), num_warps=4)
    return selected, probabilities, offsets, inverse


def _grouped_moe(x, expert_indices, router_weights, w1, w2, tokens):
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("grouped backend requires torch.nn.functional.grouped_mm")
    rows, top_k = expert_indices.shape
    experts, _, width = w1.shape
    ffn = w2.shape[-1]
    if width % 8 or ffn % 8:
        raise ValueError("native BF16 grouped GEMM requires 16-byte aligned latent and FFN widths")
    selected, probabilities, offsets, inverse = _route_plan(x, expert_indices, router_weights, experts)
    projected = F.grouped_mm(selected, w1.transpose(1, 2), offs=offsets)
    hidden = torch.empty((rows * top_k, ffn), dtype=x.dtype, device=x.device)
    long = tokens >= 2048  # long prefill chunks: one SwiGLU block per row, one gather program per token
    block = triton.next_power_of_2(ffn) if long else 256
    _weighted_swiglu_kernel[(rows * top_k, triton.cdiv(ffn, block))](
        projected, probabilities, hidden, ffn, projected.stride(0), block)
    routed = F.grouped_mm(hidden, w2.transpose(1, 2), offs=offsets)
    output = torch.empty_like(x)
    reduce = _gather_reduce_rows[(rows,)] if long else _gather_reduce[(rows, triton.cdiv(width, 128))]
    reduce(routed, inverse, output, width, top_k, routed.stride(0), triton.next_power_of_2(top_k), 128)
    return output


def _triton_moe(x, expert_indices, router_weights, w1, w2, tokens):
    rows, top_k = expert_indices.shape
    experts, _, width = w1.shape
    ffn = w2.shape[-1]
    selected, probabilities, offsets, inverse = _route_plan(x, expert_indices, router_weights, experts)
    # (BM, BN, BK, stages) from an H200 sweep; small batches favour wide N tiles.
    bm, bn, bk, stages = (16, 64, 128, 2) if tokens <= 64 else (16, 32, 128, 4) if tokens <= 128 else (64, 64, 64, 3)
    hidden = torch.empty((rows * top_k, ffn), dtype=x.dtype, device=x.device)
    _moe_up_swiglu[(experts, triton.cdiv(ffn, bn))](selected, w1, probabilities, offsets, hidden, width, ffn,
                                                    bm, bn, bk, num_warps=4, num_stages=stages)
    routed = torch.empty((rows * top_k, width), dtype=x.dtype, device=x.device)
    _moe_down[(experts, triton.cdiv(width, bn))](hidden, w2, offsets, routed, ffn, width, bm, bn, bk,
                                                 num_warps=4, num_stages=stages)
    output = torch.empty_like(x)
    _gather_reduce[(rows, triton.cdiv(width, 128))](routed, inverse, output, width, top_k, routed.stride(0),
                                                    triton.next_power_of_2(top_k), 128)
    return output


def _decode_moe(x, expert_indices, router_weights, w1, w2):
    tokens, top_k = expert_indices.shape
    width, ffn = x.shape[1], w2.shape[-1]
    hidden = torch.empty((tokens * top_k, ffn), dtype=x.dtype, device=x.device)
    routed = torch.empty((tokens * top_k, width), dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    _decode_up[(tokens * top_k, triton.cdiv(ffn, 16))](
        x, expert_indices, router_weights, w1, hidden, width, ffn, top_k,
        triton.next_power_of_2(width), 16, num_warps=4, enable_fp_fusion=False)
    _decode_down[(tokens * top_k, triton.cdiv(width, 16))](
        hidden, expert_indices, w2, routed, width, ffn, triton.next_power_of_2(ffn), 16,
        num_warps=4, enable_fp_fusion=False)
    _decode_reduce[(tokens, triton.cdiv(width, 128))](
        routed, output, width, top_k, triton.next_power_of_2(top_k), 128)
    return output


def optimized_moe(x, expert_indices, router_weights, w1, w2, *, backend="auto", decode_threshold=8, tokens=None):
    """Evaluate selected experts without modifying or renormalizing routing.

    ``auto`` uses Triton GEMV for <=8 tokens, the Triton grouped kernel for 9-256 and native grouped GEMM
    above. ``tokens`` is the per-branch token count when branches are stacked into one call (default: all
    rows), so kernel choices, and hence arithmetic, match the per-branch call exactly.
    This crossover is provisional; report and benchmark both explicit backends
    on the target GPU before claiming optimal inference throughput. Operands
    must be contiguous and CUDA production operands must be BF16. Routing IDs
    must be in [0,E); validity is the router's contract, not a synchronizing
    hot-path check. CPU execution is possible only with backend='reference'.
    """
    _validate(x, expert_indices, router_weights, w1, w2)
    if backend not in ("auto", "grouped", "triton", "decode", "reference"):
        raise ValueError(f"unknown MoE backend: {backend}")
    if backend == "reference":
        return moe_reference(x, expert_indices, router_weights, w1, w2)
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError("production MoE backends require CUDA BF16; use reference for CPU checks")
    if router_weights.dtype != torch.float32:
        raise TypeError("native Wide routing probabilities must remain FP32")
    if triton is None:
        raise RuntimeError("production MoE backends require Triton")
    if x.shape[0] == 0:
        return torch.empty_like(x)
    tokens = x.shape[0] if tokens is None else tokens
    if backend == "auto":
        backend = "decode" if tokens <= decode_threshold else "triton" if tokens <= 256 else "grouped"
    if backend == "decode":
        return _decode_moe(x, expert_indices, router_weights, w1, w2)
    if backend == "triton":
        return _triton_moe(x, expert_indices, router_weights, w1, w2, tokens)
    return _grouped_moe(x, expert_indices, router_weights, w1, w2, tokens)


routed_moe = optimized_moe
