# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""The explicit, versioned serving contract for the trained wide LatentMoE models."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WideConfig:
    variant: str = "standard"
    hidden_size: int = 1408
    num_hidden_layers: int = 18
    num_attention_heads: int = 22
    num_key_value_heads: int = 11
    head_dim: int = 64
    latent_size: int = 704
    intermediate_size: int = 704
    num_experts: int = 64
    num_experts_per_tok: int = 8
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    router_renorm_sum: float = 2.5
    prelude: int = 4
    core: int = 5
    coda: int = 4
    branch_execution: str = "serial"
    moe_backend: str = "auto"
    dtype: str = "bfloat16"
    format_version: int = 1

    def __post_init__(self):
        if self.variant not in ("standard", "concat"):
            raise ValueError("variant must be standard or concat")
        if self.branch_execution not in ("serial", "streams", "distributed_serial"):
            raise ValueError("unknown branch execution mode")
        if self.num_attention_heads != 2 * self.num_key_value_heads:
            raise ValueError("wide serving requires GQA2")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal query heads times head dimension")
        if self.head_dim % 2:
            raise ValueError("half-rotate RoPE requires an even head dimension")
        if not 0 < self.num_experts_per_tok < self.num_experts:
            raise ValueError("QB top-k must be smaller than the expert count")
        if self.variant == "concat" and self.prelude + 2 * self.core + self.coda != self.num_hidden_layers:
            raise ValueError("Concat split does not cover all physical layers")
        if self.dtype not in ("bfloat16", "float32") or self.format_version != 1:
            raise ValueError("unsupported dtype or format version")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        # Unknown keys are deliberately rejected: no silent architecture defaults.
        return cls(**value)


@dataclass(frozen=True)
class ParallelContext:
    rank: int = 0
    world_size: int = 1
    group: object = None

    def __post_init__(self):
        if self.world_size not in (1, 2, 4, 8) or not 0 <= self.rank < self.world_size:
            raise ValueError("wide serving supports world sizes 1, 2, 4, and 8")

    def bounds(self, size: int):
        return size * self.rank // self.world_size, size * (self.rank + 1) // self.world_size

    @property
    def branch_world_size(self):
        return max(1, self.world_size // 2)

    @property
    def branch_owner(self):
        return self.rank // self.branch_world_size

    def branch_context(self, group=None):
        return ParallelContext(self.rank % self.branch_world_size, self.branch_world_size, group)

    def branch_rank_groups(self):
        """Parent-group ranks; both branches share rank zero in the one-GPU control."""
        if self.world_size == 1:
            return ((0,), (0,))
        half = self.branch_world_size
        return (tuple(range(half)), tuple(range(half, self.world_size)))
