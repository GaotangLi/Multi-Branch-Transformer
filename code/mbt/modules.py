"""Small MBT modules. Importable with PyTorch alone, including on CPU."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class CheckpointModule(nn.Module):
    """A container that preserves the native shard hooks of all its children."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import sharded_state_dict_recursive
        return sharded_state_dict_recursive(
            self, prefix, sharded_offsets, metadata, getattr(self, "tp_group", None)
        )


def mark_dense_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.allreduce = True
        parameter.sequence_parallel = False


class FP32RMSNorm(nn.Module):
    """Learned RMSNorm with FP32 statistics and input-dtype output."""

    def __init__(self, width: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden.dtype)


class WidthBridge(CheckpointModule):
    """Normalize/project the common input once, before cloning branches."""

    def __init__(self, input_width: int, branch_width: int, *, initializer,
                 device=None, dtype=None, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        self.norm = FP32RMSNorm(input_width, device=device, dtype=dtype)
        self.projection = nn.Linear(input_width, branch_width, bias=False, device=device, dtype=dtype)
        initializer(self.projection.weight)
        mark_dense_parameters(self)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden))


class ConcatFusion(CheckpointModule):
    """Normalize each branch, concatenate, and project. No extra residual."""

    def __init__(self, branch_width: int, output_width: int, branches: int, *,
                 initializer, device=None, dtype=None, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        self.norms = nn.ModuleList(
            FP32RMSNorm(branch_width, device=device, dtype=dtype) for _ in range(branches)
        )
        self.fusion_projection = nn.Linear(
            branches*branch_width, output_width, bias=False, device=device, dtype=dtype
        )
        initializer(self.fusion_projection.weight)
        mark_dense_parameters(self)

    def forward(self, *states: torch.Tensor) -> torch.Tensor:
        if len(states) != len(self.norms):
            raise ValueError("fusion input count differs from the configured branch count")
        if any(state.shape != states[0].shape for state in states):
            raise ValueError("branch outputs must have matching shapes")
        return self.fusion_projection(torch.cat(
            [norm(state) for norm, state in zip(self.norms, states)], dim=-1
        ))


class JointKVProjection(CheckpointModule):
    """One shared K/V projection for one aligned branch depth.

    Inputs have already been normalized with their own attention prenorms.
    Each branch keeps its own query, gate, output projection, and MoE.
    """

    def __init__(self, width: int, branches: int, *, initializer, key_norm=None,
                 device=None, dtype=None, tp_group=None):
        super().__init__()
        if width % 128:
            raise ValueError("Joint-KV uses GQA2 with head width 64")
        self.tp_group = tp_group
        self.branches = branches
        self.groups = width // 128
        self.head_dim = 64
        self.kv_projection = nn.Linear(branches*width, width, bias=False, device=device, dtype=dtype)
        initializer(self.kv_projection.weight)
        self.key_norm = key_norm if key_norm is not None else FP32RMSNorm(64, device=device, dtype=dtype)
        mark_dense_parameters(self)

    def forward(self, *normalized_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(normalized_inputs) != self.branches:
            raise ValueError("Joint-KV input count differs from branch count")
        first = normalized_inputs[0]
        if any(value.shape != first.shape for value in normalized_inputs):
            raise ValueError("Joint-KV inputs must have matching shapes")
        joint = torch.cat(normalized_inputs, dim=-1) * (1.0/math.sqrt(self.branches))
        projected = self.kv_projection(joint).view(*first.shape[:2], self.groups, 2*self.head_dim)
        key, value = projected.split(self.head_dim, dim=-1)
        return self.key_norm(key), value.contiguous()


class BranchStack(CheckpointModule):
    """An independently parameterized stack of complete Transformer blocks."""

    def __init__(self, blocks: Iterable[nn.Module], *, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden: torch.Tensor, *, attention_mask=None, rotary_pos_emb=None):
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        return hidden
