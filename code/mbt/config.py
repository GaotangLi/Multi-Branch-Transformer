"""Portable model contracts and analytic accounting; no GPU imports.

``d`` is residual width, ``m`` the shared MoE latent width, and ``f`` each
expert's SwiGLU inner width. Counts include tied embeddings once. FLOPs are
analytic matrix FLOPs, not measured latency or hardware throughput.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

VOCABULARY = 128256
LADDER = {"W1": (12, 1024, 512), "W2": (14, 1152, 576),
          "W3": (16, 1280, 640), "W4": (18, 1408, 704)}


@dataclass(frozen=True)
class ModelConfig:
    scale: str = "W1"
    family: str = "standard"
    prelude: int | None = None
    branches: tuple[int, ...] = ()
    coda: int | None = None
    branch_hidden_size: int | None = None
    branch_latent_size: int | None = None
    branch_ffn_hidden_size: int | None = None
    kv_mode: str = "independent"
    hidden_size: int | None = None
    latent_size: int | None = None
    ffn_hidden_size: int | None = None
    num_layers: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "branches", tuple(self.branches))
        if self.scale not in LADDER:
            raise ValueError("scale must be W1, W2, W3, or W4")
        if self.family not in ("standard", "wide-shallow", "narrow-deep", "balanced"):
            raise ValueError("unknown model family")
        if self.kv_mode not in ("independent", "joint"):
            raise ValueError("kv_mode must be independent or joint")
        for key in ("hidden_size", "latent_size", "ffn_hidden_size", "num_layers",
                    "branch_hidden_size", "branch_latent_size", "branch_ffn_hidden_size"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be a positive integer")
        d, m, f = self.outer_shape
        if d % 128 or m != d // 2:
            raise ValueError("outer backbone requires GQA2/head64 and m=d/2")
        if self.family == "standard":
            if self.branches or self.kv_mode != "independent":
                raise ValueError("Standard has no parallel branches or shared KV")
            if self.prelude not in (None, 0) or self.coda not in (None, 0):
                raise ValueError("Standard uses num_layers, not prelude/coda")
            if any(getattr(self, k) is not None for k in (
                    "branch_hidden_size", "branch_latent_size", "branch_ffn_hidden_size")):
                raise ValueError("Standard has no branch dimensions")
        else:
            if len(self.branches) not in (2, 3, 4):
                raise ValueError("this release supports 2, 3, or 4 branches")
            if any(type(v) is not int or v <= 0 for v in self.branches):
                raise ValueError("branch depths must be positive integers")
            if any(type(v) is not int or v < 0 for v in (self.prelude, self.coda)):
                raise ValueError("prelude and coda must be nonnegative integers")
            if self.num_layers is not None and self.num_layers != self.physical_layers:
                raise ValueError("num_layers must equal prelude + sum(branches) + coda")
            if self.branch_shape[0] % 128:
                raise ValueError("branch residual width must be divisible by 128")
            if self.family == "wide-shallow" and self.branch_shape != self.outer_shape:
                raise ValueError("Wide-Shallow retains the outer block dimensions")
            if self.kv_mode == "joint" and len(set(self.branches)) != 1:
                raise ValueError("Joint-KV requires aligned, equal-depth branches")
            if self.family in ("narrow-deep", "balanced") and not self.uses_bridge:
                raise ValueError("Narrow-Deep/Balanced require a distinct branch width")

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        if not isinstance(value, dict):
            raise TypeError("model configuration must be an object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["branches"] = list(self.branches)
        return result

    @property
    def outer_shape(self) -> tuple[int, int, int]:
        _, base_d, base_f = LADDER[self.scale]
        d = self.hidden_size or base_d
        return d, self.latent_size or d // 2, self.ffn_hidden_size or (base_f if self.hidden_size is None else d // 2)

    @property
    def branch_shape(self) -> tuple[int, int, int]:
        d, m, f = self.outer_shape
        return (self.branch_hidden_size or d, self.branch_latent_size or m,
                self.branch_ffn_hidden_size or f)

    @property
    def physical_layers(self) -> int:
        if self.family == "standard":
            return self.num_layers or LADDER[self.scale][0]
        return self.prelude + sum(self.branches) + self.coda

    @property
    def depth(self) -> int:
        return self.physical_layers if not self.branches else self.prelude + max(self.branches) + self.coda

    @property
    def uses_bridge(self) -> bool:
        return bool(self.branches) and self.branch_shape[0] != self.outer_shape[0]

    @property
    def layer_shapes(self) -> list[tuple[int, int, int]]:
        if not self.branches:
            return [self.outer_shape] * self.physical_layers
        return ([self.outer_shape] * self.prelude + [self.branch_shape] * sum(self.branches)
                + [self.outer_shape] * self.coda)

    def counts(self, ep: int = 1) -> dict:
        if type(ep) is not int or ep not in (1, 2, 4, 8):
            raise ValueError("EP must be 1, 2, 4, or 8; TP=PP=CP=1")
        d = self.outer_shape[0]
        common = sum(4*x*x + 2*x*m + 66*x + 128 for x, m, f in self.layer_shapes) + d
        experts = sum(3*m*f for x, m, f in self.layer_shapes)
        k, w = len(self.branches), self.branch_shape[0]
        extra = (k*d*w + k*w) if k else 0
        if self.uses_bridge:
            extra += d*w + d
        delta = -self.branches[0]*(k-1)*64 if self.kv_mode == "joint" else 0
        baseline_cache = 2*sum(x for x, _, _ in self.layer_shapes)
        cache = baseline_cache
        if self.kv_mode == "joint":
            cache = 2*((self.prelude+self.coda)*d + self.branches[0]*w)
        return dict(physical=common+64*experts+VOCABULARY*d+extra+delta,
                    active_body=common+8*experts+extra+delta,
                    resident=common+(64//ep)*experts+VOCABULARY*d+extra+delta,
                    expert_bank=64*experts, vocabulary=VOCABULARY*d,
                    tied_embedding_head_parameters=VOCABULARY*d,
                    bridge_or_fusion_parameters=extra,
                    parameter_delta_vs_baseline=delta,
                    shared_kv_parameters=(self.branches[0]*(k*w*w+64) if self.kv_mode == "joint" else 0),
                    removed_branch_kv_parameters=(sum(self.branches)*(w*w+64) if self.kv_mode == "joint" else 0),
                    cache_bytes_per_token_bf16=cache,
                    baseline_cache_bytes_per_token_bf16=baseline_cache,
                    kv_cache_capacity_reduction=1-cache/baseline_cache)

    def accounting(self, tokens: int | float, sequence: int = 4096) -> dict:
        if tokens <= 0 or type(sequence) is not int or sequence <= 0:
            raise ValueError("tokens and sequence must be positive")
        d, k, w = self.outer_shape[0], len(self.branches), self.branch_shape[0]
        body = sum(6*(4*x*x+2*x*m+64*x+24*m*f+x*(sequence+1)) for x,m,f in self.layer_shapes)
        body += 6*(k+int(self.uses_bridge))*d*w
        head = 6*VOCABULARY*d
        return dict(tokens=tokens, body_train_flops_per_token=body,
                    total_train_flops_per_token=body+head,
                    body_pflop_days=body*tokens/(86400*10**15),
                    total_pflop_days=(body+head)*tokens/(86400*10**15),
                    convention="FMA=2; training matrix FLOPs=3F; causal attention S+1; excludes norms, elementwise ops, optimizer, recompute and communication")

    def architecture(self) -> dict:
        d, m, f = self.outer_shape
        k, w, layers = len(self.branches), self.branch_shape[0], self.physical_layers
        a = dict(rung=self.scale, architecture_version=2, family=self.family,
            num_layers=layers, physical_blocks=layers, hidden_size=d,
            num_attention_heads=d//64, num_query_groups=d//128, kv_channels=64,
            moe_latent_size=m, moe_ffn_hidden_size=f, num_moe_experts=64,
            moe_router_topk=8, normalization="RMSNorm", layernorm_epsilon=1e-5,
            qk_layernorm=True, qk_l2_norm=False, attention_output_gate=True,
            attention_output_norm=False, position_embedding_type="rope",
            rotary_base=500000, rotary_percent=1.0, vocabulary=VOCABULARY,
            tied_embeddings=True,
            initialization_reference_depth=layers, base_initialization_std=.02,
            output_initialization_std=.02/math.sqrt(2*layers),
            fusion_initialization_std=.02/math.sqrt(2*layers),
            prelude=self.prelude or 0, coda=self.coda or 0, branch_count=k,
            core_depths=list(self.branches), placement_split=([self.prelude,*self.branches,self.coda] if k else []),
            graph_critical_path_blocks=self.depth,
            layer_shapes=[dict(d=x,m=y,f=z,q_heads=x//64,kv_heads=x//128) for x,y,z in self.layer_shapes],
            fusion="normalized_concat" if k else None, fusion_residual=False,
            fusion_extra_moe=False, shared_input_bridge=self.uses_bridge,
            input_bridge_count=int(self.uses_bridge),
            checkpoint_layout="mbt-hierarchical-v2", kv_mode=self.kv_mode,
            parallel_cuda_execution=False, cached_decode_supported=False)
        a.update(self.counts())
        return a

    def backend_args(self) -> list[str]:
        a = self.architecture()
        pairs = {key: a[key] for key in ("num_layers", "hidden_size", "num_attention_heads",
                 "num_query_groups", "kv_channels", "moe_latent_size", "moe_ffn_hidden_size")}
        pairs.update(ffn_hidden_size=4*a["hidden_size"], num_experts=64, moe_router_topk=8,
            normalization="RMSNorm", norm_epsilon=1e-5, position_embedding_type="rope",
            rotary_base=500000, vocab_size=VOCABULARY, init_method_std=.02,
            attention_dropout=0., hidden_dropout=0., transformer_impl="transformer_engine",
            attention_backend="flash", moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0., moe_router_score_function="sigmoid", moe_router_dtype="fp32",
            moe_router_bias_update_algo="quantile", moe_router_qb_histogram_bins=10000,
            moe_router_qb_renorm_sum=2.5, moe_token_dispatcher_type="alltoall",
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, expert_tensor_parallel_size=1)
        result = [v for key,value in pairs.items() for v in ("--"+key.replace("_","-"), str(value))]
        result += ["--group-query-attention", "--qk-layernorm", "--attention-output-gate",
                   "--disable-bias-linear", "--swiglu", "--bf16", "--moe-grouped-gemm",
                   "--moe-router-enable-expert-bias", "--moe-permute-fusion", "--use-flash-attn",
                   "--no-create-attention-mask-in-dataloader"]
        return result
