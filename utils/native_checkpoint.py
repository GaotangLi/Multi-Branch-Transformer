"""Load published native model-only checkpoints with the public MBT backend.

Run full-sequence inference on one CUDA GPU using torchrun. These weights are
not optimizer checkpoints, Transformers AutoModel bundles, or cached-serving
artifacts. No private training source or dataset is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


FORMATS = {"mbt-native-shared-kv-v1", "mbt-native-standard-global-v1"}
STANDARD_LAYER_NAMES = (
    "self_attention.linear_qkv.layer_norm_weight", "self_attention.linear_qkv.weight",
    "self_attention.q_layernorm.weight", "self_attention.k_layernorm.weight",
    "self_attention.linear_proj.weight", "pre_mlp_layernorm.weight", "mlp.router.weight",
    "mlp.router.expert_bias", "mlp.fc1_latent_proj.weight", "mlp.fc2_latent_proj.weight",
    "mlp.experts.experts.linear_fc1.weight", "mlp.experts.experts.linear_fc2.weight",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_public_model():
    # Works from a source checkout or an installed public mbt package.
    code = Path(__file__).resolve().parents[1] / "code"
    if code.is_dir() and str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from mbt.config import ModelConfig
    return ModelConfig


def model_config_from_bundle(config):
    """Validate the release architecture and adapt it to public ModelConfig."""
    ModelConfig = import_public_model()
    require(config.get("bundle_format") in FORMATS, "Unsupported native bundle format")
    require(config.get("format_version") == 1, "Unsupported bundle version")
    require(config.get("dtype") == "bfloat16", "The native release requires BF16 weights")
    a = config["architecture"]
    fixed = dict(kv_channels=64, num_moe_experts=64, moe_router_topk=8,
                 layernorm_epsilon=1e-5, normalization="RMSNorm", qk_layernorm=True,
                 qk_l2_norm=False, attention_output_gate=True, attention_output_norm=False,
                 rotary_base=500000, rotary_percent=1.0, tied_embeddings=True, vocabulary=128256)
    for key, expected in fixed.items():
        require(a.get(key) == expected, f"Unsupported architecture {key}={a.get(key)!r}")
    d, m, f = (a[key] for key in ("hidden_size", "moe_latent_size", "moe_ffn_hidden_size"))
    require(a.get("num_attention_heads") == d // 64 and a.get("num_query_groups") == d // 128,
            "Only the published GQA2/head64 geometry is supported")
    common = dict(scale=a["rung"], hidden_size=d, latent_size=m,
                  ffn_hidden_size=f, num_layers=a["num_layers"])
    if config["bundle_format"] == "mbt-native-standard-global-v1":
        require(not a.get("parallel_streams"), "Standard cannot have parallel branches")
        result = ModelConfig(**common, family="standard")
    else:
        require(a.get("kv_mode") == "joint-branch-kv", "This release loader supports final Joint-KV MBTs")
        require(a.get("shared_kv_source") == "aligned_branch_block_inputs", "Unexpected shared-KV source")
        branches = tuple(a["core_depths"])
        require(branches == tuple(a.get("core_blocks", branches)), "Conflicting branch depths")
        require(len(branches) == a["branch_count"], "Conflicting branch count")
        require(len(set(branches)) == 1, "Joint-KV requires equal branch depths")
        require(a.get("shared_kv_cross_depth") is False and a.get("shared_kv_detached") is False,
                "Unsupported shared-KV depth/gradient semantics")
        require(math.isclose(a["shared_kv_input_scale"], 1 / math.sqrt(len(branches)), rel_tol=1e-12),
                "Unexpected Joint-KV input scaling")
        shapes = a["layer_shapes"]
        require(len(shapes) == a["num_layers"], "Wrong number of layer shapes")
        branch = shapes[a["prelude"]]
        result = ModelConfig(**common, family="balanced", prelude=a["prelude"],
            coda=a["coda"], branches=branches, branch_hidden_size=branch["d"],
            branch_latent_size=branch["m"], branch_ffn_hidden_size=branch["f"], kv_mode="joint")
        require(bool(a.get("shared_input_bridge")) == result.uses_bridge, "Width-bridge mismatch")
        require(a.get("fusion") == "normalized_concat", "Unsupported fusion")
        for index, (shape, expected) in enumerate(zip(shapes, result.layer_shapes, strict=True)):
            require((shape["d"], shape["m"], shape["f"]) == expected, f"Layer {index} dimensions differ")
            require(shape["q_heads"] == expected[0] // 64 and shape["kv_heads"] == expected[0] // 128,
                    f"Layer {index} attention geometry differs")
    require(result.physical_layers == a.get("physical_blocks", a["num_layers"]), "Block count mismatch")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standard_source_shapes(architecture):
    a = architecture
    d, m, f, layers, experts = (a[key] for key in
        ("hidden_size", "moe_latent_size", "moe_ffn_hidden_size", "num_layers", "num_moe_experts"))
    layer_shapes = ((d,), (3*d, d), (64,), (64,), (d, d), (d,), (experts, d),
                    (experts,), (m, d), (d, m), (experts, 2*f, m), (experts, m, f))
    result = {"embedding.word_embeddings.weight": (a["vocabulary"], d),
              "decoder.final_layernorm.weight": (d,)}
    result.update({"decoder.layers." + name: (layers, *shape)
                   for name, shape in zip(STANDARD_LAYER_NAMES, layer_shapes, strict=True)})
    return result


def weight_mapping(source_keys, target_keys, config, architecture, bundle_format):
    """Return destination -> (source key, leading indices), with no tensor copies."""
    from mbt.model import convert_legacy_state_dict
    source_keys, target_keys = set(source_keys), set(target_keys)
    if bundle_format == "mbt-native-shared-kv-v1":
        renamed = convert_legacy_state_dict({key: key for key in source_keys}, config)
        mapping = {destination: (source, ()) for destination, source in renamed.items()}
    else:
        require(source_keys == set(standard_source_shapes(architecture)), "Unexpected Standard global tensor keys")
        mapping = {}
        for source in sorted(source_keys):
            if not source.startswith("decoder.layers."):
                mapping[source] = (source, ())
                continue
            suffix = source.removeprefix("decoder.layers.")
            for layer in range(config.physical_layers):
                if suffix.startswith("mlp.experts.experts."):
                    fc = suffix.split(".")[-2]
                    legacy = f"decoder.layers.{layer}.mlp.experts.{fc}.weight"
                    destination = next(iter(convert_legacy_state_dict({legacy: None}, config)))
                    if destination in target_keys:
                        mapping[destination] = (source, (layer,))
                    else:
                        for expert in range(architecture["num_moe_experts"]):
                            legacy_expert = legacy + str(expert)
                            destination = next(iter(convert_legacy_state_dict({legacy_expert: None}, config)))
                            mapping[destination] = (source, (layer, expert))
                else:
                    legacy = f"decoder.layers.{layer}.{suffix}"
                    destination = next(iter(convert_legacy_state_dict({legacy: None}, config)))
                    mapping[destination] = (source, (layer,))
    require(set(mapping) == target_keys,
            "Native tensor mapping mismatch; missing=" + repr(sorted(target_keys - set(mapping)))
            + "; unexpected=" + repr(sorted(set(mapping) - target_keys)))
    require({source for source, _ in mapping.values()} == source_keys, "Unconsumed source tensors")
    return mapping


def load_model(bundle, *, max_sequence_length=4096):
    """Construct and strictly load the public EP1 model; caller initializes NCCL/MP.

    Returns ``(model, receipt)``. The model accepts ``input_ids, position_ids,
    attention_mask=None`` and returns full-sequence logits [batch, tokens, vocab].
    ``manifest.loader_sha256`` is historical export provenance, not a hash of
    this new adapter; current config and model-weight checksums are verified.
    """
    import torch
    from safetensors import safe_open
    directory = Path(bundle).resolve(strict=True)
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    model_config = model_config_from_bundle(config)
    require(manifest.get("bundle_format") == config["bundle_format"], "Manifest format mismatch")
    require(manifest.get("optimizer_exported") is False, "Expected a model-only bundle")
    for filename, key in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
        require(sha256(directory / filename) == manifest[key], "Checksum mismatch: " + filename)
    from mbt.model import build_model, make_native_config
    native = make_native_config(model_config, ep=1, use_cpu_initialization=False,
        perform_initialization=True, gradient_accumulation_fusion=False,
        moe_permute_fusion=True, bias_activation_fusion=True, bias_dropout_fusion=True,
        masked_softmax_fusion=True, apply_rope_fusion=True, persist_layer_norm=True)
    from megatron.core import parallel_state
    require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1,
            "Initialize a one-GPU process group first (or use the torchrun CLI)")
    require(parallel_state.model_parallel_is_initialized(), "Initialize Megatron model parallelism first")
    for getter in (parallel_state.get_tensor_model_parallel_world_size,
                   parallel_state.get_pipeline_model_parallel_world_size,
                   parallel_state.get_expert_model_parallel_world_size,
                   parallel_state.get_context_parallel_world_size):
        require(getter() == 1, "Native model-only loading requires TP=PP=EP=CP=1")
    model = build_model(model_config, native_config=native,
                        max_sequence_length=max_sequence_length).cuda().bfloat16().eval()
    for layer in model.decoder.layers:
        layer.mlp.router._maintain_float32_expert_bias()
    state = model.state_dict(keep_vars=True)
    # Preserve constructor-generated TE non-FP8 metadata only. No model tensor
    # is allowed to retain a random initialization or be silently ignored.
    extra_state_keys = [key for key in state if key.endswith("._extra_state")]
    target = {key: value for key, value in state.items() if key not in extra_state_keys}
    require(all(torch.is_tensor(value) for value in target.values()), "Unexpected non-tensor model state")
    require(model.shared_embedding_or_output_weight() is model.embedding.word_embeddings.weight,
            "Embedding and output head must be tied")
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        mapping = weight_mapping(keys, target, model_config, config["architecture"], config["bundle_format"])
        expected_global = standard_source_shapes(config["architecture"]) if config["bundle_format"].endswith("standard-global-v1") else None
        source_elements = 0
        for name in keys:
            view = source.get_slice(name)
            shape = tuple(view.get_shape())
            if expected_global is not None:
                require(shape == expected_global[name], "Unexpected global shape: " + name)
            source_elements += math.prod(shape)
        target_bytes = 0
        for name, (source_name, indices) in mapping.items():
            view = source.get_slice(source_name)
            shape = tuple(view.get_shape())
            require(all(0 <= index < shape[axis] for axis, index in enumerate(indices)), "Invalid tensor slice")
            require(tuple(target[name].shape) == shape[len(indices):], "Tensor shape mismatch: " + name)
            expected_dtype = torch.float32 if name.endswith("expert_bias") else torch.bfloat16
            require(target[name].dtype == expected_dtype and view.get_dtype() == ("F32" if expected_dtype == torch.float32 else "BF16"),
                    "Tensor dtype mismatch: " + name)
            target_bytes += target[name].numel() * target[name].element_size()
        require(sum(value.numel() for value in target.values()) == source_elements,
                "Mapped tensor coverage does not match the complete source payload")
        require(target_bytes == manifest["model_bytes"], "Model byte count mismatch")
        with torch.no_grad():
            for name, (source_name, indices) in mapping.items():
                value = source.get_tensor(source_name)
                if indices:
                    value = value[indices]
                require(bool(torch.isfinite(value).all()), "Nonfinite tensor: " + source_name)
                target[name].copy_(value)
                require(torch.equal(target[name], value.to(target[name].device)), "Loaded tensor differs: " + name)
    receipt = dict(status="PASS", exact_tensor_load=True, bundle_format=config["bundle_format"],
        source_tensors=len(keys), loaded_tensors=len(target), model_bytes=target_bytes,
        constructor_te_extra_state_entries=len(extra_state_keys), fp32_router_bias=True,
        tied_embedding_head=True, cached_decode_supported=False, optimizer_loaded=False,
        model_config=model_config.to_dict())
    model.native_bundle_receipt = receipt
    return model, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--token-ids", default="128000,791,3925,315,8198,369,264,2115")
    parser.add_argument("--inspect", action="store_true", help="Print architecture only; no weights/GPU imports")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = json.loads((args.bundle / "config.json").read_text())
    model_config = model_config_from_bundle(config)
    if args.inspect:
        print(json.dumps(model_config.to_dict(), indent=2))
        return
    import torch
    from mbt.model import _prepare_backend
    _prepare_backend()
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    require(int(os.environ.get("WORLD_SIZE", "0")) == 1, "Use torchrun --standalone --nproc_per_node=1")
    ids = [int(value.strip()) for value in args.token_ids.split(",")]
    require(1 <= len(ids) <= 4096 and all(0 <= value < 128256 for value in ids), "Invalid token-ID fixture")
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, expert_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        model, receipt = load_model(args.bundle)
        tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
        positions = torch.arange(len(ids), device="cuda")[None]
        with torch.inference_mode():
            logits = model(tokens, positions, None)
        require(tuple(logits.shape) == (1, len(ids), 128256), "Unexpected logits shape")
        require(bool(torch.isfinite(logits).all()), "Nonfinite full-sequence logits")
        receipt.update(logits_shape=list(logits.shape), logits_dtype=str(logits.dtype),
                       last_token_top_ids=logits[0, -1].topk(5).indices.tolist())
        print(json.dumps(receipt, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
