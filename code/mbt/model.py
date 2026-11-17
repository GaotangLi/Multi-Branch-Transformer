"""Lightweight model factory and explicit legacy weight-key conversion.

Importing this module does not import PyTorch or initialize a GPU. Native
dependencies are loaded only when constructing the model/configuration.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib.util
import math
from pathlib import Path
import sys

from .config import ModelConfig, VOCABULARY


def backend_path() -> Path:
    sibling = Path(__file__).resolve().parent.parent / "backend"
    if sibling.is_dir():
        return sibling
    spec = importlib.util.find_spec("mbt_backend")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    raise FileNotFoundError("native Megatron backend is not installed")


def _prepare_backend(root=None) -> Path:
    root = (Path(root) if root is not None else backend_path()).resolve()
    if not (root / "megatron/core/transformer/moe/moe_layer.py").is_file():
        raise FileNotFoundError(f"MBT native backend is missing at {root}")
    already_loaded = sys.modules.get("megatron.core")
    if already_loaded is not None:
        filename = getattr(already_loaded, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError("a different Megatron installation is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _fixed_native_fields(model_config: ModelConfig) -> dict:
    architecture = model_config.architecture()
    fields = {key: architecture[key] for key in (
        "num_layers", "hidden_size", "num_attention_heads", "num_query_groups", "kv_channels",
        "moe_latent_size", "moe_ffn_hidden_size", "num_moe_experts", "moe_router_topk",
        "normalization", "layernorm_epsilon", "qk_layernorm", "qk_l2_norm",
        "attention_output_gate", "attention_output_norm",
    )}
    fields.update(ffn_hidden_size=4*architecture["hidden_size"],
        moe_layer_freq=1, gated_linear_unit=True, add_bias_linear=False, add_qkv_bias=False,
        hidden_dropout=0., attention_dropout=0., init_method_std=.02,
        layernorm_zero_centered_gamma=False, multi_latent_attention=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_tensor_parallel_size=1,
        sequence_parallel=False, bf16=True, fp16=False, cpu_offloading=False,
        moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.,
        moe_router_enable_expert_bias=True, moe_router_bias_update_algo="quantile",
        moe_router_qb_histogram_bins=10000, moe_router_qb_renorm_sum=2.5,
        moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        moe_router_pre_softmax=False, moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True, transformer_impl="transformer_engine",
        fused_single_qkv_rope=False, apply_query_key_layer_scaling=False,
        fp32_residual_connection=False, use_mup=False)
    return fields


def make_native_config(model_config: ModelConfig, *, ep: int = 1, backend_root=None, **overrides):
    """Construct the documented native recipe, useful for small GPU tests.

    Training normally supplies ``core_transformer_config_from_args`` instead.
    Overrides are checked again by ``build_model`` before model construction.
    """
    _prepare_backend(backend_root)
    import torch
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    values = _fixed_native_fields(model_config)
    values.update(params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        activation_func=torch.nn.functional.silu, expert_model_parallel_size=ep,
        attention_backend=AttnBackend.flash)
    values.update(overrides)
    return TransformerConfig(**values)


def validate_native_config(model_config: ModelConfig, native_config) -> None:
    """Reject backend settings outside the public model contract."""
    import torch
    mismatches = [f"{key}={getattr(native_config, key, None)!r}; expected {expected!r}"
        for key, expected in _fixed_native_fields(model_config).items()
        if getattr(native_config, key, None) != expected]
    if native_config.params_dtype != torch.bfloat16:
        mismatches.append("params_dtype must be torch.bfloat16")
    if native_config.activation_func is not torch.nn.functional.silu:
        mismatches.append("latent experts require SwiGLU (activation_func=silu)")
    if native_config.expert_model_parallel_size not in (1, 2, 4, 8):
        mismatches.append("expert parallel size must be 1, 2, 4, or 8")
    for key in ("window_size", "fp8", "fp4", "output_soft_cap", "moe_z_loss_coeff",
                "moe_shared_expert_intermediate_size", "moe_expert_capacity_factor"):
        if getattr(native_config, key, None) is not None:
            mismatches.append(f"{key} is outside the released recipe")
    if getattr(native_config, "recompute_granularity", None) not in (None, "selective"):
        mismatches.append("only selective core-attention recomputation is supported")
    if getattr(native_config, "recompute_granularity", None) == "selective" and set(
            getattr(native_config, "recompute_modules", ()) or ()) != {"core_attn"}:
        mismatches.append("selective recomputation must use only core_attn")
    if getattr(native_config, "cuda_graph_impl", "none") not in (None, "none"):
        mismatches.append("CUDA graph capture is not supported by the training release")
    if any(getattr(native_config, key, False) for key in (
            "fine_grained_activation_offloading", "defer_embedding_wgrad_compute", "delay_wgrad_compute")):
        mismatches.append("activation offloading and deferred weight gradients are unsupported")
    initializers = (("init_method", .02),
        ("output_layer_init_method", .02/math.sqrt(2*model_config.physical_layers)))
    for name, expected in initializers:
        actual = getattr(getattr(native_config, name, None), "keywords", {}).get("std", -1)
        if not math.isclose(actual, expected, rel_tol=1e-12):
            mismatches.append(f"{name} must use normal std={expected}")
    if mismatches:
        raise ValueError("MBT native configuration mismatch: " + "; ".join(mismatches))


def build_model(model_config: ModelConfig, *, native_config, backend_root=None,
                vocab_size=VOCABULARY, max_sequence_length=4096, pre_process=True,
                post_process=True, parallel_output=True, pg_collection=None):
    """Build an explicit MBT model without modifying any backend classes."""
    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a ModelConfig")
    if vocab_size != VOCABULARY:
        raise ValueError(f"the released tokenizer/model contract uses {VOCABULARY} tokens")
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be a positive integer")
    _prepare_backend(backend_root)
    validate_native_config(model_config, native_config)
    from .modeling import MBTModel
    return MBTModel(model_config, native_config, vocab_size=vocab_size,
        max_sequence_length=max_sequence_length, pre_process=pre_process,
        post_process=post_process, parallel_output=parallel_output, pg_collection=pg_collection)


def _layer_prefixes(config: ModelConfig) -> list[str]:
    if not config.branches:
        return [f"decoder.prelude.{index}." for index in range(config.physical_layers)]
    return ([f"decoder.prelude.{index}." for index in range(config.prelude)]
        + [f"decoder.branches.{branch}.blocks.{depth}."
           for branch, length in enumerate(config.branches) for depth in range(length)]
        + [f"decoder.coda.{index}." for index in range(config.coda)])


def convert_legacy_state_dict(state_dict, config: ModelConfig, *, reverse: bool = False):
    """Rename raw model weight keys between the historical and explicit graph.

    Values/tensors are not copied or changed. This is a weights-only conversion,
    not an optimizer-state or distributed-checkpoint migration. A caller must
    load the result strictly into the matching model configuration.
    """
    prefixes = _layer_prefixes(config)
    mapping = {f"decoder.layers.{index}.": prefix for index, prefix in enumerate(prefixes)}
    mapping.update({f"decoder.stream_concat_fusion.norm_{letter}.": f"decoder.fusion.norms.{index}."
                    for index, letter in enumerate("abcd")})
    mapping.update({"decoder.stream_concat_fusion.fusion_projection.": "decoder.fusion.fusion_projection.",
                    "decoder.shared_kv_memory.banks.": "decoder.joint_kv."})
    if reverse:
        mapping = {value: key for key, value in mapping.items()}
    ordered = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
    result = OrderedDict()
    for key, value in state_dict.items():
        converted = key
        for old, new in ordered:
            if key.startswith(old):
                converted = new + key[len(old):]
                break
        if converted in result:
            raise ValueError(f"weight-key conversion collision at {converted}")
        result[converted] = value
    return result
