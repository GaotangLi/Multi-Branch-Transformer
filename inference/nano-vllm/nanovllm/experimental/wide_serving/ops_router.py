# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Optional, inference-only QB router optimizations for a future source version.

Explicit opt-in model backends import this module. Preserve FP32 F.linear and native
torch.topk(K+1, sorted=True), including native tie behavior. The optional
"fused" postprocessing fuses gather+sigmoid and normalization+ID-copy around
the SAME Torch sum operation, preserving its reduction ordering. The separate
"fused_post8" opt-in replaces all three postprocessing kernels for the exactly
qualified E64/K8/2.5 shape; its reduction identity must be requalified on other
stacks. Neither path replaces top-k's tie rule or biases the probabilities.

The low-risk first option is an explicit FP32 weight cache with post_backend
"torch". The caller must rebuild the cache after loading, moving, or updating
weights, keep it FP32, and recapture any graph that used the old cache.
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = libdevice = None


def prepare_router_weight(weight):
    """Return an independent, detached, contiguous FP32 snapshot of [E,D].

    This function is a load-time operation, not a forward-path cache lookup.
    A nonpersistent module buffer is a suitable owner once integration is
    authorized. Do not subsequently pass that buffer through a BF16 cast.
    """
    if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("router weight must be a 2D BF16 or FP32 tensor")
    if not weight.is_contiguous():
        raise ValueError("pack router weights once at load time before creating the cache")
    return weight.detach().to(dtype=torch.float32, copy=True).contiguous()


def _check(logits, selected, topk, renorm_sum):
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("router logits must be FP32 [tokens,experts]")
    if not 0 < topk < logits.shape[1]:
        raise ValueError("topk must be positive and smaller than the expert count")
    if selected.shape != (logits.shape[0], topk + 1) or selected.dtype != torch.int64:
        raise ValueError("selected IDs must be native int64 torch.topk(K+1) output")
    if selected.device != logits.device:
        raise ValueError("logits and selected IDs must share a device")
    if not isinstance(renorm_sum, (int, float)) or not math.isfinite(renorm_sum) or renorm_sum <= 0:
        raise ValueError("renorm_sum must be a positive finite scalar")


def postprocess_reference(logits, selected, topk, renorm_sum=2.5):
    """Exact existing eager Torch probability expression (CPU or CUDA)."""
    _check(logits, selected, topk, renorm_sum)
    indices = selected[:, :topk]
    probabilities = torch.sigmoid(logits.gather(-1, indices))
    probabilities = probabilities * (renorm_sum / (probabilities.sum(-1, keepdim=True) + 1e-9))
    return indices.contiguous(), probabilities.contiguous()


def route_reference(x, weight, expert_bias, topk=8, renorm_sum=2.5):
    """The unchanged model router, including its per-call weight conversion."""
    logits = F.linear(x.float(), weight.float())
    selected = torch.topk(logits + expert_bias.float(), topk + 1, dim=-1, sorted=True).indices
    return postprocess_reference(logits, selected, topk, renorm_sum)


if triton is not None:

    @triton.jit
    def _gather_sigmoid(Logits, Selected, Probabilities,
                        LOGIT_ROW: tl.constexpr, LOGIT_COL: tl.constexpr,
                        INDEX_ROW: tl.constexpr, INDEX_COL: tl.constexpr,
                        K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        expert = tl.load(Selected + row * INDEX_ROW + col * INDEX_COL, col < K, other=0)
        value = tl.load(Logits + row * LOGIT_ROW + expert * LOGIT_COL, col < K, other=0)
        # Match PyTorch's real FP32 sigmoid: one/(one+std::exp(-value)).
        probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-value))
        tl.store(Probabilities + row * K + col, probability, col < K)

    @triton.jit
    def _normalize_and_copy(Probabilities, Sums, Selected, OutProb, OutIds,
                            INDEX_ROW: tl.constexpr, INDEX_COL: tl.constexpr,
                            K: tl.constexpr, RENORM: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        probability = tl.load(Probabilities + row * K + col, col < K, other=0)
        denominator = tl.load(Sums + row) + 1e-9
        # Tensor.__rtruediv__ in installed Torch is reciprocal() * scalar.
        # Keep these as separate FP32 operations rather than RENORM/denominator.
        reciprocal = tl.div_rn(1.0, denominator)
        scale = reciprocal * RENORM
        expert = tl.load(Selected + row * INDEX_ROW + col * INDEX_COL, col < K, other=0)
        tl.store(OutProb + row * K + col, probability * scale, col < K)
        tl.store(OutIds + row * K + col, expert, col < K)

    @triton.jit
    def _post8(Logits, Selected, OutProb, OutIds, E: tl.constexpr):
        # Floating-point body from qualified router_post_fusion.py; wide addressing.
        # Exactness of this reduction tree is stack-specific, unlike keeping
        # Torch sum in fused_postprocess. Keep the existing backend available.
        row = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, 8)
        expert = tl.load(Selected + row*9 + k)
        value = tl.load(Logits + row*E + expert)
        probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-value))
        denominator = tl.sum(probability, 0) + 1e-9
        scale = tl.div_rn(1.0, denominator) * 2.5
        tl.store(OutProb + row*8 + k, probability*scale)
        tl.store(OutIds + row*8 + k, expert)

    @triton.jit
    def _post8_stacked(Logits, Selected, OutProb, OutIds, T: tl.constexpr, E: tl.constexpr):
        # _post8 on stacked branches; ids are offset by E * branch to index the branches' concatenated experts.
        row = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, 8)
        expert = tl.load(Selected + row*9 + k)
        value = tl.load(Logits + row*E + expert)
        probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-value))
        denominator = tl.sum(probability, 0) + 1e-9
        scale = tl.div_rn(1.0, denominator) * 2.5
        tl.store(OutProb + row*8 + k, probability*scale)
        tl.store(OutIds + row*8 + k, expert + (row // T)*E)


def fused_postprocess(logits, selected, topk, renorm_sum=2.5):
    """Fuse post-selection work while preserving Torch's row-sum operation.

    Native top-k IDs are copied unchanged. Probabilities may still differ by
    a few FP32 ULPs across CUDA math implementations; qualification reports
    this explicitly, including downstream BF16 rounding sensitivity.
    """
    _check(logits, selected, topk, renorm_sum)
    if logits.device.type != "cuda" or triton is None:
        raise RuntimeError("fused router postprocessing requires CUDA and Triton")
    tokens = logits.shape[0]
    ids = torch.empty((tokens, topk), device=logits.device, dtype=torch.int64)
    output = torch.empty((tokens, topk), device=logits.device, dtype=torch.float32)
    if not tokens:
        return ids, output
    raw = torch.empty_like(output)
    block = triton.next_power_of_2(topk)
    _gather_sigmoid[(tokens,)](
        logits, selected, raw, logits.stride(0), logits.stride(1), selected.stride(0), selected.stride(1),
        topk, block, num_warps=4, enable_fp_fusion=False)
    # Exact same shape, layout, dtype and operation as the eager expression.
    sums = raw.sum(-1, keepdim=True)
    _normalize_and_copy[(tokens,)](
        raw, sums, selected, output, ids, selected.stride(0), selected.stride(1),
        topk, float(renorm_sum), block, num_warps=4, enable_fp_fusion=False)
    return ids, output


def fused_postprocess8(logits, selected, topk=8, renorm_sum=2.5):
    """One post-top9 kernel for the E64/K8 router contract.

    Selection remains native sorted Torch top9; probabilities use UNBIASED
    logits and the first eight selected IDs. No new top-k/tie algorithm.
    """
    _check(logits, selected, topk, renorm_sum)
    if (logits.shape[1] != 64 or topk != 8 or renorm_sum != 2.5
            or not logits.is_contiguous() or not selected.is_contiguous()):
        raise ValueError('fused_post8 requires contiguous E64/K8/top9 and renorm_sum=2.5')
    if not logits.is_cuda or triton is None:
        raise RuntimeError('fused_post8 requires CUDA and Triton')
    tokens = logits.shape[0]
    ids = torch.empty((tokens, 8), device=logits.device, dtype=torch.int64)
    output = torch.empty((tokens, 8), device=logits.device, dtype=torch.float32)
    if tokens:
        _post8[(tokens,)](logits, selected, output, ids, 64, num_warps=4, enable_fp_fusion=False)
    return ids, output


def fused_postprocess8_stacked(logits, selected, tokens_per_branch):
    """fused_postprocess8 for [n*T, 64] stacked branch logits; returns ids into the n*64 concatenated experts."""
    rows = selected.shape[0]
    ids = torch.empty((rows, 8), device=logits.device, dtype=torch.int64)
    output = torch.empty((rows, 8), device=logits.device, dtype=torch.float32)
    if rows:
        _post8_stacked[(rows,)](logits, selected, output, ids, tokens_per_branch, 64, num_warps=4, enable_fp_fusion=False)
    return ids, output


def cached_qb_route(x, weight_fp32, expert_bias, topk=8, renorm_sum=2.5, *, post_backend="torch"):
    """FP32 router with an explicit load-time weight cache; no host tensor reads.

    "torch" is the exact cached-weight control and also supports CPU testing.
    "triton" changes only post-selection operations; full-model qualification
    remains required before adoption, especially near routing/rounding margins.
    """
    if x.ndim != 2 or weight_fp32.ndim != 2 or x.shape[1] != weight_fp32.shape[1]:
        raise ValueError("expected x[tokens,width], cached weight[experts,width]")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight_fp32.dtype != torch.float32:
        raise ValueError("input must be BF16/FP32 and cached router weight must remain FP32")
    if expert_bias.shape != (weight_fp32.shape[0],) or expert_bias.dtype != torch.float32:
        raise ValueError("persistent expert bias must be FP32 [experts]")
    if x.device != weight_fp32.device or x.device != expert_bias.device:
        raise ValueError("router operands must share a device")
    if not weight_fp32.is_contiguous():
        raise ValueError("cached weight must be contiguous")
    if post_backend not in ("torch", "triton", "triton_post8"):
        raise ValueError("post_backend must be torch, triton, or triton_post8")
    if post_backend == 'triton_post8' and (weight_fp32.shape[0] != 64 or topk != 8 or renorm_sum != 2.5):
        raise ValueError('fused_post8 requires E64/K8 and renorm_sum=2.5')
    if not 0 < topk < weight_fp32.shape[0]:
        raise ValueError("topk must be positive and smaller than the expert count")
    logits = F.linear(x.float(), weight_fp32)
    selected = torch.topk(logits + expert_bias, topk + 1, dim=-1, sorted=True).indices
    post = {'torch': postprocess_reference, 'triton': fused_postprocess,
            'triton_post8': fused_postprocess8}[post_backend]
    return post(logits, selected, topk, renorm_sum)
