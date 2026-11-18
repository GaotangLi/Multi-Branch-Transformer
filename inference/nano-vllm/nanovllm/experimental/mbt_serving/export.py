# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Strict model-only export of trusted native Megatron DCP checkpoints.

Unlike HF AutoModel checkpoints, these bundles use the explicit MBT serving
schema. Native DCP metadata is pickle: inspect/export ONLY trusted checkpoints.
All tensor loading, hashing and serialization belongs on allocated compute.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re

from .config import MBTConfig


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
    "mlp.experts.experts.linear_fc1.weight": "moe.w1",
    "mlp.experts.experts.linear_fc2.weight": "moe.w2",
}
DIRECT_MAP = {
    "embedding.word_embeddings.weight": "embedding.weight",
    "decoder.final_layernorm.weight": "norm.weight",
}


def layer_shapes(config, index):
    c = config.layer_config(index)
    d, h, m, f, e = (c.hidden_size, c.head_dim, c.latent_size,
                      c.intermediate_size, c.num_experts)
    result = {
        "input_norm.weight": (d,), "attention.qkv.weight": (3*d, d),
        "attention.q_norm.weight": (h,), "attention.k_norm.weight": (h,),
        "attention.out.weight": (d, d), "post_attention_norm.weight": (d,),
        "moe.router.weight": (e, d), "moe.router.expert_bias": (e,),
        "moe.down.weight": (m, d), "moe.up.weight": (d, m),
        "moe.w1": (e, 2*f, m), "moe.w2": (e, m, f),
    }
    if config.kv_mode != "independent" and config.layer_role(index).startswith("branch_"):
        result["attention.qkv.weight"] = (2*d, d)  # [Q, gate] rows; K/V and the K norm live in a bank
        del result["attention.k_norm.weight"]
    return result


def direct_map(config):
    result = dict(DIRECT_MAP)
    if config.variant == "mbt":
        for branch in range(len(config.branch_depths)):
            result[f"decoder.stream_concat_fusion.norm_{'abcd'[branch]}.weight"] = (
                f"fusion.norms.{branch}.weight")
        result["decoder.stream_concat_fusion.fusion_projection.weight"] = "fusion.projection.weight"
    if config.uses_input_bridge:
        result["decoder.width_in.norm.weight"] = "input_bridge.norm.weight"
        result["decoder.width_in.projection.weight"] = "input_bridge.projection.weight"
    return result


def canonical_shapes(config):
    d = config.hidden_size
    result = {"embedding.weight": (config.vocab_size, d), "norm.weight": (d,)}
    for index in range(config.num_hidden_layers):
        result.update({f"layers.{index}.{key}": shape
                       for key, shape in layer_shapes(config, index).items()})
    if config.variant == "mbt":
        w, k = config.branch_hidden_size, len(config.branch_depths)
        result.update({f"fusion.norms.{i}.weight": (w,) for i in range(k)})
        result["fusion.projection.weight"] = (d, k*w)
    if config.uses_input_bridge:
        result["input_bridge.norm.weight"] = (d,)
        result["input_bridge.projection.weight"] = (config.branch_hidden_size, d)
    for bank in range(config.kv_banks):
        result[f"kv_banks.{bank}.kv_projection.weight"] = (config.branch_hidden_size, config.bank_input_width)
        result[f"kv_banks.{bank}.key_norm.weight"] = (config.head_dim,)
        if config.kv_mode == "prelude-kv":
            result[f"kv_banks.{bank}.input_norm.weight"] = (config.bank_input_width,)
    return result


def native_shapes(config, layout):
    """Both native DCP forms: a stacked layer axis, or explicit physical keys."""
    target = canonical_shapes(config)
    result = {native: target[canonical] for native, canonical in direct_map(config).items()}
    if layout == "stacked":
        first = layer_shapes(config, 0)
        if any(layer_shapes(config, i) != first for i in range(config.num_hidden_layers)):
            raise ValueError("heterogeneous layers cannot use a single stacked checkpoint axis")
        result.update({"decoder.layers." + native: (config.num_hidden_layers, *first[canonical])
                       for native, canonical in LAYER_MAP.items()})
    elif layout == "explicit":
        for index in range(config.num_hidden_layers):
            result.update({f"decoder.layers.{index}." + native:
                           target[f"layers.{index}." + canonical]
                           for native, canonical in LAYER_MAP.items()})
    else:
        raise ValueError("unknown native layer-key layout")
    return result


def expected_dtype(name, config):
    return "float32" if name.endswith("expert_bias") else config.dtype


def validate_metadata(metadata, config):
    """Shape-only validation, usable with tiny descriptor fixtures and no Torch."""
    entries = metadata.state_dict_metadata
    model = {name: item for name, item in entries.items()
             if name.startswith(("decoder.", "embedding.", "output_layer."))
             and hasattr(item, "size")}
    explicit = any(re.match(r"decoder\.layers\.\d+\.", name) for name in model)
    layout = "explicit" if explicit else "stacked"
    expected = native_shapes(config, layout)
    if set(model) != set(expected):
        raise ValueError(f"native model key mismatch: missing={sorted(set(expected)-set(model))}; "
                         f"unexpected={sorted(set(model)-set(expected))}")
    size_bytes = 0
    for name, shape in expected.items():
        item = model[name]
        dtype = str(item.properties.dtype).removeprefix("torch.")
        if tuple(item.size) != shape or dtype != expected_dtype(name, config):
            raise ValueError(f"native model shape/dtype mismatch: {name}")
        size_bytes += math.prod(shape) * (4 if dtype == "float32" else 2)
    return {"native_layout": layout, "native_tensor_count": len(expected),
            "canonical_tensor_count": len(canonical_shapes(config)), "model_bytes": size_bytes}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(bundle, expected_format):
    """Bind a converted export to the exact native bundle it came from."""
    bundle = Path(bundle)
    declared = json.loads((bundle / "manifest.json").read_text()).get("bundle_format")
    if declared != expected_format:
        raise ValueError(f"bundle declares {declared!r}, not {expected_format!r}")
    return {"source_bundle_format": expected_format,
            "source_name": json.loads((bundle / "config.json").read_text()).get("name"),
            "source_config_sha256": file_sha256(bundle / "config.json"),
            "source_weights_sha256": file_sha256(bundle / "model.safetensors")}


def inspect_checkpoint(checkpoint, config, *, trusted=False):
    if not trusted:
        raise ValueError("native .metadata is pickle: explicit trusted=True is required")
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Deliberately never load common.pt or optimizer/RNG tensors.
    with (checkpoint / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    return dict(validate_metadata(metadata, config),
                metadata_sha256=file_sha256(checkpoint / ".metadata"))


def verify_model_contract(model, config):
    """Bind requested inference graph to the completed native proof, not a name."""
    fixed = {
        "hidden_size": config.hidden_size, "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_query_groups": config.num_key_value_heads, "kv_channels": config.head_dim,
        "moe_latent_size": config.latent_size, "moe_ffn_hidden_size": config.intermediate_size,
        "num_moe_experts": config.num_experts, "moe_router_topk": config.num_experts_per_tok,
        "rotary_base": config.rope_theta, "vocabulary": config.vocab_size,
        "layernorm_epsilon": config.rms_norm_eps, "normalization": "RMSNorm",
        "attention": "full_causal_gqa2", "attention_output_gate": True,
        "attention_output_norm": False, "qk_layernorm": True, "qk_l2_norm": False,
        "qk_norm_location": "pre_rope", "tied_embeddings": True,
        "loop_repeats": 1, "loss_count": 1, "prefix": False,
        "gate_type": "query_dependent_channel_sigmoid", "gate_multiplier": 1.0,
        "gate_location": "post_sdpa_pre_output_projection", "position_embedding_type": "rope",
        "parallel_streams": config.variant == "mbt",
    }
    # Older proofs omit these constants; only the original qualified recipe is
    # exportable until the native proof starts recording alternative values.
    if config.router_renorm_sum != 2.5 or config.max_position_embeddings != 4096:
        raise ValueError("native export supports only QB renorm 2.5 and training context 4096")
    for key, value in fixed.items():
        if model.get(key) != value:
            raise ValueError(f"formal proof architecture mismatch: {key}={model.get(key)!r}, expected {value!r}")
    if config.variant == "mbt":
        for key, value in {"prelude": config.prelude, "coda": config.coda,
                           "fusion_residual": False, "fusion_extra_moe": False,
                           "fusion_norm_epsilon": config.rms_norm_eps,
                           "fusion_norms_learned": True}.items():
            if model.get(key) != value:
                raise ValueError(f"formal proof branch/fusion mismatch: {key}")
        if model.get("fusion") not in ("concat", "normalized_concat"):
            raise ValueError("only pure normalized concat fusion is supported")
        depths = model.get("core_depths", model.get("core_blocks"))
        if depths is None:
            depths = [model.get("core_a"), model.get("core_b")]
        if tuple(depths) != tuple(config.branch_depths):
            raise ValueError("formal proof branch depths mismatch")
        if model.get("branch_count", len(depths)) != len(config.branch_depths):
            raise ValueError("formal proof branch count mismatch")
        bridge = model.get("shared_input_bridge", bool(model.get("input_bridge_count", 0)))
        if bridge != config.uses_input_bridge:
            raise ValueError("formal proof shared input bridge mismatch")
        shape = [config.hidden_size, len(config.branch_depths)*config.branch_hidden_size]
        if model.get("fusion_projection_shape") != shape:
            raise ValueError("formal proof fusion projection shape mismatch")
        if config.uses_input_bridge:
            if model.get("input_bridge_shape") != [config.branch_hidden_size, config.hidden_size]:
                raise ValueError("formal proof input bridge shape mismatch")
            if model.get("input_bridge_count") != 1 or model.get("bridge_residual") is not False:
                raise ValueError("formal proof requires exactly one non-residual shared bridge")
    expected = []
    for index in range(config.num_hidden_layers):
        c = config.layer_config(index)
        expected.append(dict(d=c.hidden_size, m=c.latent_size, f=c.intermediate_size,
                             q_heads=c.num_attention_heads, kv_heads=c.num_key_value_heads))
    if "layer_shapes" in model:
        if model["layer_shapes"] != expected:
            raise ValueError("formal proof per-layer widths mismatch")
    elif any(shape != expected[0] for shape in expected):
        raise ValueError("heterogeneous export requires explicit native per-layer shape evidence")


def verify_proof(proof_path, checkpoint, config):
    proof_path = Path(proof_path).resolve(strict=True)
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("stage") != "formal":
        raise ValueError("export requires a completed formal checkpoint proof")
    if proof.get("iteration") != proof.get("horizon") or "100" not in proof.get("labels", []):
        raise ValueError("only the completed full-data checkpoint is supported")
    checkpoint = Path(checkpoint).resolve(strict=True)
    if Path(proof["checkpoint"]).resolve(strict=True) != checkpoint:
        raise ValueError("formal proof binds another checkpoint")
    verify_model_contract(proof["model"], config)
    inventory = proof["checkpoint_inventory"]
    if any(Path(name).name != name or name in (".", "..") for name in inventory):
        raise ValueError("invalid proof checkpoint member name")
    actual_files = {item.name for item in checkpoint.iterdir() if item.is_file()}
    if actual_files != set(inventory):
        raise ValueError("checkpoint inventory changed since its completion proof")
    for name, info in inventory.items():
        path = checkpoint / name
        if path.is_symlink():
            raise ValueError("checkpoint member symlinks are not accepted")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (info["size_bytes"], info["mtime_ns"]):
            raise ValueError("checkpoint changed after completion: " + name)
    hashes = proof["metadata_sha256"]
    if ".metadata" not in hashes or not set(hashes) <= set(inventory):
        raise ValueError("formal proof has incomplete/invalid metadata hashes")
    for name, digest in hashes.items():
        if file_sha256(checkpoint / name) != digest:
            raise ValueError("checkpoint metadata checksum mismatch: " + name)
    source_digest = proof.get("source_inventory_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ValueError("formal proof lacks a source-inventory SHA256 binding")
    # This allowlist is public-safe: never copy private paths or raw proof JSON.
    return {"proof_sha256": file_sha256(proof_path), "iteration": proof["iteration"],
            "source_inventory_sha256": source_digest}


def convert_native_tensors(native, config, layout):
    """Keep native [Q,gate,K,V] and SwiGLU row ordering; no math conversion."""
    import torch
    expected = native_shapes(config, layout)
    if set(native) != set(expected):
        raise ValueError("native tensor set differs from export schema")
    for name, shape in expected.items():
        value = native[name]
        if tuple(value.shape) != shape or value.dtype != getattr(torch, expected_dtype(name, config)):
            raise ValueError("native tensor shape/dtype mismatch: " + name)
    result = {canonical: native[name].contiguous() for name, canonical in direct_map(config).items()}
    for index in range(config.num_hidden_layers):
        for native_name, canonical in LAYER_MAP.items():
            name = "decoder.layers." + (f"{index}." if layout == "explicit" else "") + native_name
            value = native[name] if layout == "explicit" else native[name][index]
            result[f"layers.{index}.{canonical}"] = value.contiguous()
    return result


def nearest_existing(target):
    return next(parent for parent in (target, *target.parents) if parent.exists())


def scratch_destination(path, scratch_root=None):
    """Resolve an export destination. An export is always a new directory and never overwrites or
    merges into an existing one. A declared scratch root additionally confines the export to that
    root and keeps it off any home filesystem, for sites where home is too small for model weights.
    """
    target, root = Path(path).resolve(), None
    if scratch_root is not None:
        root = Path(scratch_root).resolve(strict=True)
        if not root.is_dir() or len(root.parts) < 3:
            raise ValueError("select a personal large-scratch root, never a filesystem root")
        if any(root == home or root.is_relative_to(home) for home in
               (Path.home().resolve(), Path("/home"))):
            raise ValueError("home filesystem is forbidden for model exports")
        if target == root or not target.is_relative_to(root):
            raise ValueError("output must be a new child directory within declared scratch")
    if target.exists():
        raise FileExistsError("destination exists; exports never overwrite or merge")
    if not os.access(root if root is not None else nearest_existing(target), os.W_OK | os.X_OK):
        raise PermissionError("export destination is not writable")
    return target


def export_checkpoint(checkpoint, proof, output, config, scratch_root=None, *,
                      trusted=False):
    target = scratch_destination(output, scratch_root)
    provenance = verify_proof(proof, checkpoint, config)
    report = inspect_checkpoint(checkpoint, config, trusted=trusted)
    usage = os.statvfs(Path(scratch_root) if scratch_root is not None else nearest_existing(target))
    if usage.f_bavail * usage.f_frsize < 2*report["model_bytes"] + 1024**3:
        raise RuntimeError("insufficient shared free space for weights and serialization margin")
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    native = {name: torch.empty(shape, dtype=getattr(torch, expected_dtype(name, config)), device="cpu")
              for name, shape in native_shapes(config, report["native_layout"]).items()}
    load(native, storage_reader=FileSystemReader(str(checkpoint)),
         planner=DefaultLoadPlanner(flatten_state_dict=False))
    converted = convert_native_tensors(native, config, report["native_layout"])
    if any(not bool(value.isfinite().all()) for value in converted.values()):
        raise ValueError("checkpoint contains a non-finite model tensor")
    verify_proof(proof, checkpoint, config)
    target.mkdir(parents=True, exist_ok=False)
    # If serialization fails, the incomplete directory is intentionally retained
    # for diagnosis; no manifest means it cannot be loaded as a valid bundle.
    save_file(converted, str(target / "model.safetensors"), metadata={"format": "mbt-serving-v2"})
    export_config = replace(config, branch_execution="serial", moe_backend="auto")
    (target / "config.json").write_text(json.dumps(export_config.to_dict(), indent=2) + "\n")
    manifest = dict(report, **provenance, format_version=2, bundle_format="mbt-serving-v2",
                    optimizer_exported=False, runtime_parity_qualified=False,
                    config_sha256=file_sha256(target / "config.json"),
                    weights_sha256=file_sha256(target / "model.safetensors"),
                    storage_note="Free-space check is not an applicable quota guarantee; review site quota separately.")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_export(model, directory, verify_hashes=True):
    """Safe tensor loader with exact architecture/shape/dtype and checksum checks."""
    import torch
    from safetensors import safe_open
    model.invalidate_router_caches()
    directory = Path(directory).resolve(strict=True)
    disk = MBTConfig.from_dict(json.loads((directory / "config.json").read_text()))
    local = replace(model.config, branch_execution=disk.branch_execution, moe_backend=disk.moe_backend)
    if local != disk:
        raise ValueError("serving model architecture differs from exported config")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != 2 or manifest.get("bundle_format") != "mbt-serving-v2":
        raise ValueError("unsupported model-bundle format")
    if file_sha256(directory / "config.json") != manifest["config_sha256"]:
        raise ValueError("export config checksum mismatch")
    if verify_hashes and file_sha256(directory / "model.safetensors") != manifest["weights_sha256"]:
        raise ValueError("export weights checksum mismatch")
    expected = canonical_shapes(disk)
    destination = model.state_dict(keep_vars=True)
    if set(destination) != set(expected):
        raise ValueError(f"runtime/export key mismatch: missing={sorted(set(expected)-set(destination))}; "
                         f"unexpected={sorted(set(destination)-set(expected))}")
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as stream:
        if set(stream.keys()) != set(expected):
            raise ValueError("export has missing or unexpected tensor keys")
        # Validate the complete file BEFORE changing any live model tensors.
        for name, shape in expected.items():
            view = stream.get_slice(name)
            dtype = "F32" if expected_dtype(name, disk) == "float32" else "BF16"
            if tuple(view.get_shape()) != shape or view.get_dtype() != dtype:
                raise ValueError("export tensor shape/dtype mismatch: " + name)
            if tuple(destination[name].shape) != shape or destination[name].dtype != getattr(torch, expected_dtype(name, disk)):
                raise ValueError("runtime tensor shape/dtype mismatch: " + name)
        with torch.no_grad():
            for name in expected:
                value = stream.get_tensor(name)
                dest = destination[name]
                if dest.shape != value.shape or dest.dtype != value.dtype:
                    raise ValueError("runtime tensor shape/dtype mismatch: " + name)
                dest.copy_(value)
    model.prepare_router_caches()
    return {"canonical_tensors": len(expected), "loaded_local_tensors": len(destination),
            "weights_checksum_verified": bool(verify_hashes), "format_version": 2,
            "native_parity_qualified": bool(manifest.get("runtime_parity_qualified", False))}


def export_shared_bundle(bundle, output, scratch_root=None):
    """Native shared-KV model bundle (mbt-native-shared-kv-v1: config.json, manifest.json,
    model.safetensors) -> mbt-serving-v2 export. Names are mapped and experts stacked; no math."""
    import torch
    from safetensors.torch import load_file, save_file
    bundle = Path(bundle).resolve(strict=True)
    target = scratch_destination(output, scratch_root)
    a = json.loads((bundle / "config.json").read_text())["architecture"]
    outer, core = a["layer_shapes"][0], a["layer_shapes"][a["prelude"]]
    config = MBTConfig(variant="mbt", hidden_size=a["hidden_size"], num_hidden_layers=a["num_layers"],
                       num_attention_heads=a["num_attention_heads"], num_key_value_heads=a["num_query_groups"],
                       head_dim=a["kv_channels"], latent_size=outer["m"], intermediate_size=outer["f"],
                       num_experts=a["num_moe_experts"], num_experts_per_tok=a["moe_router_topk"],
                       vocab_size=a["vocabulary"], rms_norm_eps=a["layernorm_epsilon"], rope_theta=float(a["rotary_base"]),
                       prelude=a["prelude"], coda=a["coda"], branch_depths=tuple(a["core_depths"]),
                       core_hidden_size=core["d"], core_latent_size=core["m"], core_intermediate_size=core["f"],
                       input_bridge=bool(a["shared_input_bridge"]), kv_mode=a["kv_mode"])
    native, expected = load_file(str(bundle / "model.safetensors")), canonical_shapes(config)
    result = {canonical: native.pop(name) for name, canonical in direct_map(config).items()}
    for index in range(config.num_hidden_layers):
        stem = f"decoder.layers.{index}."
        for native_name, canonical in LAYER_MAP.items():
            if f"layers.{index}.{canonical}" in expected and not native_name.startswith("mlp.experts."):
                result[f"layers.{index}.{canonical}"] = native.pop(stem + native_name)
        for name, fc in (("moe.w1", "linear_fc1"), ("moe.w2", "linear_fc2")):
            result[f"layers.{index}.{name}"] = torch.stack([native.pop(f"{stem}mlp.experts.{fc}.weight{e}")
                                                            for e in range(config.num_experts)])
    for bank in range(config.kv_banks):
        for name in ("kv_projection", "key_norm") + (("input_norm",) if config.kv_mode == "prelude-kv" else ()):
            result[f"kv_banks.{bank}.{name}.weight"] = native.pop(f"decoder.shared_kv_memory.banks.{bank}.{name}.weight")
    if native:
        raise ValueError(f"unconsumed native tensors: {sorted(native)[:8]}")
    target.mkdir(parents=True, exist_ok=False)
    save_file({name: value.contiguous() for name, value in result.items()}, str(target / "model.safetensors"),
              metadata={"format": "mbt-serving-v2"})
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    manifest = dict(format_version=2, bundle_format="mbt-serving-v2", kv_mode=config.kv_mode,
                    canonical_tensor_count=len(result), optimizer_exported=False,
                    runtime_parity_qualified=False, config_sha256=file_sha256(target / "config.json"),
                    weights_sha256=file_sha256(target / "model.safetensors"),
                    **source_provenance(bundle, "mbt-native-shared-kv-v1"))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_standard_bundle(bundle, output, scratch_root=None):
    """Native Standard model bundle (mbt-native-standard-global-v1: one tensor per role with a
    leading layer axis) -> mbt-serving-v2 export. Layers are sliced and names mapped; no math."""
    from safetensors.torch import load_file, save_file
    bundle = Path(bundle).resolve(strict=True)
    target = scratch_destination(output, scratch_root)
    a = json.loads((bundle / "config.json").read_text())["architecture"]
    config = MBTConfig(variant="standard", hidden_size=a["hidden_size"], num_hidden_layers=a["num_layers"],
                       num_attention_heads=a["num_attention_heads"], num_key_value_heads=a["num_query_groups"],
                       head_dim=a["kv_channels"], latent_size=a["moe_latent_size"],
                       intermediate_size=a["moe_ffn_hidden_size"], num_experts=a["num_moe_experts"],
                       num_experts_per_tok=a["moe_router_topk"], vocab_size=a["vocabulary"],
                       rms_norm_eps=a["layernorm_epsilon"], rope_theta=float(a["rotary_base"]))
    # convert_native_tensors validates the whole key set, every shape and every dtype first.
    result = convert_native_tensors(load_file(str(bundle / "model.safetensors")), config, "stacked")
    target.mkdir(parents=True, exist_ok=False)
    save_file(result, str(target / "model.safetensors"), metadata={"format": "mbt-serving-v2"})
    (target / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    manifest = dict(format_version=2, bundle_format="mbt-serving-v2",
                    canonical_tensor_count=len(result), optimizer_exported=False,
                    runtime_parity_qualified=False, config_sha256=file_sha256(target / "config.json"),
                    weights_sha256=file_sha256(target / "model.safetensors"),
                    **source_provenance(bundle, "mbt-native-standard-global-v1"))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def convert_native_bundle(bundle, output, scratch_root=None):
    """Convert either published native bundle to an mbt-serving-v2 export, picked by its manifest."""
    declared = json.loads((Path(bundle) / "manifest.json").read_text()).get("bundle_format")
    converters = {"mbt-native-shared-kv-v1": export_shared_bundle,
                  "mbt-native-standard-global-v1": export_standard_bundle}
    if declared not in converters:
        raise ValueError(f"no converter for bundle_format {declared!r}; "
                         f"known: {sorted(converters)}")
    return converters[declared](bundle, output, scratch_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--config", required=True, help="explicit MBTConfig v2 JSON")
    parser.add_argument("--output")
    parser.add_argument("--scratch-root", help="optional: confine the export to this root and keep it off any home filesystem")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--trusted-native-checkpoint", action="store_true",
                        help="acknowledge that native DCP pickle is trusted code-bearing input")
    args = parser.parse_args()
    config = MBTConfig.from_dict(json.loads(Path(args.config).read_text()))
    if not args.trusted_native_checkpoint:
        parser.error("--trusted-native-checkpoint is required; never inspect untrusted native pickle")
    if args.metadata_only:
        provenance = verify_proof(args.proof, args.checkpoint, config)
        report = dict(inspect_checkpoint(args.checkpoint, config, trusted=True), **provenance)
    else:
        if not args.output:
            parser.error("--output is required unless --metadata-only")
        report = export_checkpoint(args.checkpoint, args.proof, args.output, config, args.scratch_root,
                                   trusted=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
