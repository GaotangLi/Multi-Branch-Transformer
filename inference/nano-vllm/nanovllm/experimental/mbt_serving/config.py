# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Strict v2 architecture contract for portable MBT model-only exports."""

from dataclasses import asdict, dataclass

from ..wide_serving.config import WideConfig


@dataclass(frozen=True)
class MBTConfig:
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
    prelude: int = 0
    coda: int = 0
    branch_depths: tuple[int, ...] = ()
    core_hidden_size: int | None = None
    core_latent_size: int | None = None
    core_intermediate_size: int | None = None
    input_bridge: bool = False
    branch_execution: str = "serial"
    moe_backend: str = "auto"
    dtype: str = "bfloat16"
    format_version: int = 2
    # "independent": one KV cache per attention layer. Shared KV (branch layers keep only Q/gate rows):
    # "prelude-kv": one bank projected from the prelude output, read by every branch layer;
    # "joint-branch-kv": one bank per aligned depth, projected from all branches' normalized inputs.
    kv_mode: str = "independent"

    def __post_init__(self):
        object.__setattr__(self, "branch_depths", tuple(self.branch_depths))
        if self.format_version != 2 or self.variant not in ("standard", "mbt"):
            raise ValueError("v2 supports variant=standard or mbt only")
        if self.kv_mode not in ("independent", "prelude-kv", "joint-branch-kv"):
            raise ValueError("unknown kv_mode")
        allowed = ("serial", "streams") if self.kv_mode == "independent" else ("serial", "batched")
        if self.branch_execution not in allowed:
            raise ValueError(f"{self.kv_mode} KV supports single-GPU {' or '.join(allowed)} execution")
        if self.kv_mode != "independent" and not (self.variant == "mbt" and self.input_bridge):
            raise ValueError("shared-KV banks are defined for bridged MBT models only")
        if self.kv_mode == "joint-branch-kv" and len(set(self.branch_depths)) != 1:
            raise ValueError("joint-branch-kv requires equal branch depths")
        if type(self.input_bridge) is not bool:
            raise ValueError("input_bridge must be an explicit boolean")
        if self.moe_backend not in ("auto", "reference", "decode", "grouped"):
            raise ValueError("unsupported MoE backend")
        positive = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "latent_size", "intermediate_size", "vocab_size",
                    "max_position_embeddings", "num_experts", "num_experts_per_tok")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in positive):
            raise ValueError("model dimensions, counts and lengths must be positive integers")
        if any(type(value) is not int or value < 0 for value in (self.prelude, self.coda)):
            raise ValueError("prelude and coda must be nonnegative integers")
        if any(value is not None and (type(value) is not int or value <= 0) for value in
               (self.core_hidden_size, self.core_latent_size, self.core_intermediate_size)):
            raise ValueError("optional core dimensions must be positive integers")
        if not self.rms_norm_eps > 0 or not self.rope_theta > 0 or not self.router_renorm_sum > 0:
            raise ValueError("normalization, RoPE and routing constants must be positive")
        if self.variant == "standard":
            if self.branch_depths or self.prelude or self.coda or self.input_bridge or any(value is not None for value in
                    (self.core_hidden_size, self.core_latent_size, self.core_intermediate_size)):
                raise ValueError("Standard has no branch allocation, core overrides or input bridge")
            if self.branch_execution != "serial":
                raise ValueError("Standard has no parallel branch execution")
        else:
            if not 2 <= len(self.branch_depths) <= 4 or any(type(value) is not int or value < 1 for value in self.branch_depths):
                raise ValueError("MBT requires 2–4 positive branch depths, unequal depths are allowed")
            if self.physical_layer_count != self.num_hidden_layers:
                raise ValueError("prelude + sum(branch_depths) + coda must equal physical num_hidden_layers")
            if self.branch_hidden_size != self.hidden_size and not self.input_bridge:
                raise ValueError("different branch width requires the shared input_bridge")
        # This reuses precisely the existing full-attention GQA2 numerical contract.
        self._block_config(False)
        if self.variant == "mbt":
            self._block_config(True)

    @property
    def branch_hidden_size(self):
        return self.hidden_size if self.core_hidden_size is None else self.core_hidden_size

    @property
    def branch_latent_size(self):
        return self.latent_size if self.core_latent_size is None else self.core_latent_size

    @property
    def branch_intermediate_size(self):
        return self.intermediate_size if self.core_intermediate_size is None else self.core_intermediate_size

    @property
    def uses_input_bridge(self):
        return self.input_bridge

    @property
    def physical_layer_count(self):
        return self.num_hidden_layers if self.variant == "standard" else self.prelude + sum(self.branch_depths) + self.coda

    @property
    def branch_ranges(self):
        ranges, start = [], self.prelude
        for depth in self.branch_depths:
            ranges.append((start, start + depth))
            start += depth
        return tuple(ranges)

    @property
    def kv_banks(self):
        return {"independent": 0, "prelude-kv": 1, "joint-branch-kv": max(self.branch_depths, default=0)}[self.kv_mode]

    @property
    def bank_input_width(self):
        # prelude-kv: the outer-width prelude output; joint-branch-kv: concat of the K normalized branch inputs
        return self.hidden_size if self.kv_mode == "prelude-kv" else len(self.branch_depths) * self.branch_hidden_size

    @property
    def level_layers(self):
        """Per depth level, the layer IDs of the branches still active, deepest branch first."""
        order = sorted(range(len(self.branch_depths)), key=lambda b: -self.branch_depths[b])
        return tuple(tuple(self.branch_ranges[b][0] + level for b in order if self.branch_depths[b] > level)
                     for level in range(max(self.branch_depths, default=0)))

    def layer_role(self, index):
        if not 0 <= index < self.num_hidden_layers:
            raise IndexError(index)
        if self.variant == "standard":
            return "standard"
        for branch, (start, end) in enumerate(self.branch_ranges):
            if start <= index < end:
                return f"branch_{branch}"
        return "prelude" if index < self.prelude else "coda"

    def layer_config(self, index):
        return self._block_config(self.layer_role(index).startswith("branch_"))

    def _block_config(self, branch):
        d = self.branch_hidden_size if branch else self.hidden_size
        if branch:
            if d % (2 * self.head_dim):
                raise ValueError("branch width must be divisible by 2*head_dim for GQA2")
            heads, groups = d // self.head_dim, d // (2 * self.head_dim)
        else:
            heads, groups = self.num_attention_heads, self.num_key_value_heads
        return WideConfig(variant="standard", hidden_size=d, num_hidden_layers=1,
                          num_attention_heads=heads, num_key_value_heads=groups, head_dim=self.head_dim,
                          latent_size=self.branch_latent_size if branch else self.latent_size,
                          intermediate_size=self.branch_intermediate_size if branch else self.intermediate_size,
                          num_experts=self.num_experts, num_experts_per_tok=self.num_experts_per_tok,
                          vocab_size=self.vocab_size, max_position_embeddings=self.max_position_embeddings,
                          rms_norm_eps=self.rms_norm_eps, rope_theta=self.rope_theta,
                          router_renorm_sum=self.router_renorm_sum, prelude=0, core=0, coda=0,
                          branch_execution="serial", moe_backend=self.moe_backend, dtype=self.dtype)

    def to_dict(self):
        result = asdict(self)
        result["branch_depths"] = list(self.branch_depths)
        if self.kv_mode == "independent":
            del result["kv_mode"]  # independent-KV config.json stays byte-identical to v2
        return result

    @classmethod
    def from_dict(cls, value):
        # The dataclass rejects unknown keys rather than guessing architecture.
        return cls(**value)
