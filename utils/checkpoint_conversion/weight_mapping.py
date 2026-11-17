"""Exact tensor mapping for the released native MBT and Standard bundles.

This module imports PyTorch only when tensor conversion is requested. No
cluster paths, native Megatron imports, uploads, or weight downloads are used.
"""
from __future__ import annotations

import math


LAYER_MAP = {
    "self_attention.linear_qkv.layer_norm_weight": "input_norm.weight",
    "self_attention.linear_qkv.weight": "attention.qkv.weight",
    "self_attention.q_layernorm.weight": "attention.q_norm.weight",
    "self_attention.k_layernorm.weight": "attention.k_norm.weight",
    "self_attention.linear_proj.weight": "attention.out.weight",
    "pre_mlp_layernorm.weight": "post_attention_norm.weight",
    "mlp.router.weight": "moe.router.weight",
    "mlp.router.expert_bias": "moe.router.expert_bias",
    "mlp.fc1_latent_proj.weight": "moe.down.weight",
    "mlp.fc2_latent_proj.weight": "moe.up.weight",
}
DIRECT_MAP = {
    "embedding.word_embeddings.weight": "embedding.weight",
    "decoder.final_layernorm.weight": "norm.weight",
}
STANDARD_FORMAT = "mbt-native-standard-global-v1"
MBT_FORMAT = "mbt-native-shared-kv-v1"


def configuration_from_architecture(a):
    """Extract only architecture fields; do not infer dimensions from run names."""
    fixed = {
        "normalization": "RMSNorm", "qk_layernorm": True, "qk_l2_norm": False,
        "qk_norm_location": "pre_rope", "attention_output_gate": True,
        "attention_output_norm": False, "tied_embeddings": True,
        "position_embedding_type": "rope", "rotary_percent": 1.0,
        "gate_type": "query_dependent_channel_sigmoid", "gate_multiplier": 1.0,
        "gate_location": "post_sdpa_pre_output_projection",
    }
    for key, value in fixed.items():
        if a.get(key) != value:
            raise ValueError(f"Unsupported native architecture: {key}={a.get(key)!r}")
    if a.get("attention") != "full_causal_gqa2":
        raise ValueError("Only the original paper GQA2 architecture is in scope")
    result = dict(
        vocab_size=a["vocabulary"], hidden_size=a["hidden_size"],
        num_hidden_layers=a["num_layers"], num_attention_heads=a["num_attention_heads"],
        num_key_value_heads=a["num_query_groups"], head_dim=a["kv_channels"],
        moe_latent_size=a["moe_latent_size"], moe_intermediate_size=a["moe_ffn_hidden_size"],
        num_experts=a["num_moe_experts"], num_experts_per_tok=a["moe_router_topk"],
        router_renorm_sum=2.5, rms_norm_eps=a["layernorm_epsilon"],
        rope_theta=a["rotary_base"], max_position_embeddings=4096,
        tie_word_embeddings=True, bos_token_id=128000, eos_token_id=128001,
        pad_token_id=128254, branch_depths=[], prelude=0, coda=0,
        branch_hidden_size=a["hidden_size"],
        branch_moe_latent_size=a["moe_latent_size"],
        branch_moe_intermediate_size=a["moe_ffn_hidden_size"],
        branch_num_attention_heads=a["num_attention_heads"],
        branch_num_key_value_heads=a["num_query_groups"],
        kv_mode="independent", use_input_bridge=False,
    )
    if a.get("parallel_streams"):
        depths = list(a.get("core_depths", a.get("core_blocks", [])))
        if len(depths) not in (2, 3, 4) or len(set(depths)) != 1 or min(depths) <= 0:
            raise ValueError("Only the final aligned two/three/four-branch models are in scope")
        if a.get("kv_mode") != "joint-branch-kv":
            raise ValueError("Only the selected Joint-KV finals are in scope")
        p, c = a["prelude"], a["coda"]
        if p + sum(depths) + c != a["num_layers"]:
            raise ValueError("Native block allocation is inconsistent")
        shapes = a["layer_shapes"]
        if len(shapes) != a["num_layers"]:
            raise ValueError("Missing native layer shapes")
        core = shapes[p]
        outside = dict(d=a["hidden_size"], m=a["moe_latent_size"],
                       f=a["moe_ffn_hidden_size"], q_heads=a["num_attention_heads"],
                       kv_heads=a["num_query_groups"])
        if any(shape != (core if p <= i < p + sum(depths) else outside)
               for i, shape in enumerate(shapes)):
            raise ValueError("Native layers are not uniform within the outer/core groups")
        if (a.get("fusion") != "normalized_concat" or a.get("fusion_residual")
                or not a.get("fusion_norms_learned") or a.get("fusion_extra_moe")):
            raise ValueError("Expected normalized concat projection without a residual")
        bridge = a.get("shared_input_bridge")
        if bridge is not True or a.get("input_bridge_count") != 1 or a.get("bridge_residual"):
            raise ValueError("Expected one shared, nonresidual input width bridge")
        if not math.isclose(a["shared_kv_input_scale"], 1 / math.sqrt(len(depths)), abs_tol=1e-15):
            raise ValueError("Unexpected Joint-KV input scaling")
        if a.get("shared_kv_cross_depth") or a.get("shared_kv_detached"):
            raise ValueError("Unexpected native shared-KV semantics")
        result.update(
            prelude=p, coda=c, branch_depths=depths,
            branch_hidden_size=core["d"], branch_moe_latent_size=core["m"],
            branch_moe_intermediate_size=core["f"],
            branch_num_attention_heads=core["q_heads"],
            branch_num_key_value_heads=core["kv_heads"],
            kv_mode="joint", use_input_bridge=True,
        )
    elif a.get("implementation") != "native_standard_gpt_no_forward_override":
        raise ValueError("Expected the original sequential Standard reference")
    return result


def layer_shape(config, index):
    start = config["prelude"]
    branch = bool(config["branch_depths"]) and start <= index < start + sum(config["branch_depths"])
    prefix = "branch_" if branch else ""
    return dict(d=config[prefix + "hidden_size"], m=config[prefix + "moe_latent_size"],
                f=config[prefix + "moe_intermediate_size"],
                q=config[prefix + "num_attention_heads"],
                kv=config[prefix + "num_key_value_heads"],
                shared=branch and config["kv_mode"] == "joint")


def canonical_shapes(config):
    d, h, e = config["hidden_size"], config["head_dim"], config["num_experts"]
    result = {"embedding.weight": (config["vocab_size"], d), "norm.weight": (d,)}
    for index in range(config["num_hidden_layers"]):
        s = layer_shape(config, index)
        q_rows = 2 * s["q"] * h
        kv_rows = 0 if s["shared"] else 2 * s["kv"] * h
        shapes = {
            "input_norm.weight": (s["d"],),
            "attention.qkv.weight": (q_rows + kv_rows, s["d"]),
            "attention.q_norm.weight": (h,),
            "attention.out.weight": (s["d"], s["q"] * h),
            "post_attention_norm.weight": (s["d"],),
            "moe.router.weight": (e, s["d"]), "moe.router.expert_bias": (e,),
            "moe.down.weight": (s["m"], s["d"]), "moe.up.weight": (s["d"], s["m"]),
            "moe.w1": (e, 2 * s["f"], s["m"]), "moe.w2": (e, s["m"], s["f"]),
        }
        if not s["shared"]:
            shapes["attention.k_norm.weight"] = (h,)
        result.update({f"layers.{index}.{key}": shape for key, shape in shapes.items()})
    k = len(config["branch_depths"])
    if k:
        w = config["branch_hidden_size"]
        result.update({f"fusion.norms.{i}.weight": (w,) for i in range(k)})
        result["fusion.projection.weight"] = (d, k * w)
        if config["use_input_bridge"]:
            result["input_bridge.norm.weight"] = (d,)
            result["input_bridge.projection.weight"] = (w, d)
        if config["kv_mode"] == "joint":
            rows = 2 * config["branch_num_key_value_heads"] * h
            for depth in range(config["branch_depths"][0]):
                result[f"joint_kv.{depth}.projection.weight"] = (rows, k * w)
                result[f"joint_kv.{depth}.key_norm.weight"] = (h,)
    return result


def convert_values(values, config, native_format):
    """Return canonical weights and an exhaustive, reversible mapping report."""
    import torch
    if native_format not in (STANDARD_FORMAT, MBT_FORMAT):
        raise ValueError("Unknown source weight format")
    expected = canonical_shapes(config)
    consumed, result, mapping, aliases = set(), {}, [], []

    def read(name):
        if name not in values:
            raise ValueError("Missing source tensor: " + name)
        consumed.add(name)
        return values[name]

    def put(target, value, sources, operation="rename"):
        if target not in expected or target in result:
            raise ValueError("Unexpected/duplicate destination: " + target)
        dtype = torch.float32 if target.endswith("expert_bias") else torch.bfloat16
        if tuple(value.shape) != expected[target] or value.dtype != dtype:
            raise ValueError(f"Shape/dtype mismatch {target}: {tuple(value.shape)}/{value.dtype}, expected {expected[target]}/{dtype}")
        if not bool(value.isfinite().all()):
            raise ValueError("Nonfinite source weights: " + target)
        if operation.startswith("layer_axis["):
            source = values[sources[0]]
            if tuple(source.shape) != (config["num_hidden_layers"], *expected[target]):
                raise ValueError("Stacked source has an unexpected leading layer axis: " + sources[0])
        result[target] = value.detach().contiguous()
        mapping.append(dict(target="model." + target, sources=sources, operation=operation))

    for native, target in DIRECT_MAP.items():
        put(target, read(native), [native])
    # Some state_dict emit the tied LM head; discard only after exact equality.
    if "output_layer.weight" in values:
        value = read("output_layer.weight")
        embedding = values["embedding.word_embeddings.weight"]
        if value.shape != embedding.shape or value.dtype != embedding.dtype or not torch.equal(value, embedding):
            raise ValueError("Output layer is not exactly tied to the token embeddings")
        aliases.append(dict(source="output_layer.weight", target="model.embedding.weight", exact_equal=True))

    for index in range(config["num_hidden_layers"]):
        shared = layer_shape(config, index)["shared"]
        for native, canonical in LAYER_MAP.items():
            if canonical == "attention.k_norm.weight" and shared:
                continue
            if native_format == STANDARD_FORMAT:
                source = "decoder.layers." + native
                value = read(source)[index]
                operation = f"layer_axis[{index}]"
            else:
                source = f"decoder.layers.{index}." + native
                value, operation = read(source), "rename"
            put(f"layers.{index}." + canonical, value, [source], operation)
        for fc, canonical in (("linear_fc1", "moe.w1"), ("linear_fc2", "moe.w2")):
            if native_format == STANDARD_FORMAT:
                source = "decoder.layers.mlp.experts.experts." + fc + ".weight"
                value, sources, operation = read(source)[index], [source], f"layer_axis[{index}]"
            else:
                base = f"decoder.layers.{index}.mlp.experts.{fc}.weight"
                if base in values:
                    value, sources, operation = read(base), [base], "rename_grouped_experts"
                else:
                    sources = [base + str(e) for e in range(config["num_experts"])]
                    value = torch.stack([read(source) for source in sources], dim=0)
                    operation = "stack_expert_axis_in_numerical_order"
            put(f"layers.{index}." + canonical, value, sources, operation)

    k = len(config["branch_depths"])
    if k:
        for index in range(k):
            source = f"decoder.stream_concat_fusion.norm_{'abcd'[index]}.weight"
            put(f"fusion.norms.{index}.weight", read(source), [source])
        source = "decoder.stream_concat_fusion.fusion_projection.weight"
        put("fusion.projection.weight", read(source), [source])
        if config["use_input_bridge"]:
            for member in ("norm", "projection"):
                source = f"decoder.width_in.{member}.weight"
                put(f"input_bridge.{member}.weight", read(source), [source])
        for depth in range(config["branch_depths"][0]):
            for source_member, target_member in (("kv_projection", "projection"), ("key_norm", "key_norm")):
                source = f"decoder.shared_kv_memory.banks.{depth}.{source_member}.weight"
                put(f"joint_kv.{depth}.{target_member}.weight", read(source), [source])
    if set(result) != set(expected):
        raise ValueError("Incomplete destination: " + repr(sorted(set(expected) - set(result))))
    if consumed != set(values):
        raise ValueError("Unmapped source tensor(s): " + repr(sorted(set(values) - consumed)))
    destination = {"model." + name: value for name, value in result.items()}
    return destination, dict(
        source_tensor_count=len(values), consumed_source_tensor_count=len(consumed),
        target_tensor_count=len(destination), mapping=mapping, exact_tied_aliases=aliases,
        source_bytes=sum(v.numel() * v.element_size() for v in values.values()),
        target_bytes=sum(v.numel() * v.element_size() for v in destination.values()),
        excluded_nonalias_tensors=[], parameter_dtypes="BF16; persistent router bias FP32",
    )

