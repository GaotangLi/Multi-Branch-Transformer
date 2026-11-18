# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Strict, model-only conversion from the native Megatron DCP checkpoint.

The metadata inspection path reads no tensor payload. Conversion is an explicit
allocation task and never restores optimizer, RNG, dataloader, or TE extra state.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle

import torch

from .config import WideConfig


_LAYER_MAP = {
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
_DIRECT_MAP = {
    "embedding.word_embeddings.weight": "embedding.weight",
    "decoder.final_layernorm.weight": "norm.weight",
}
_FUSION_MAP = {
    "decoder.stream_concat_fusion.norm_a.weight": "fusion.norm_a.weight",
    "decoder.stream_concat_fusion.norm_b.weight": "fusion.norm_b.weight",
    "decoder.stream_concat_fusion.fusion_projection.weight": "fusion.projection.weight",
}


def canonical_shapes(config):
    d, h, m, f, e = (config.hidden_size, config.head_dim, config.latent_size,
                      config.intermediate_size, config.num_experts)
    layer = {
        "input_norm.weight": (d,), "attention.qkv.weight": (3*d, d),
        "attention.q_norm.weight": (h,), "attention.k_norm.weight": (h,),
        "attention.out.weight": (d, d), "post_attention_norm.weight": (d,),
        "moe.router.weight": (e, d), "moe.router.expert_bias": (e,),
        "moe.down.weight": (m, d), "moe.up.weight": (d, m),
        "moe.w1": (e, 2*f, m), "moe.w2": (e, m, f),
    }
    result = {"embedding.weight": (config.vocab_size, d), "norm.weight": (d,)}
    for index in range(config.num_hidden_layers):
        result.update({f"layers.{index}.{name}": shape for name, shape in layer.items()})
    if config.variant == "concat":
        result.update({"fusion.norm_a.weight": (d,), "fusion.norm_b.weight": (d,),
                       "fusion.projection.weight": (d, 2*d)})
    return result


def native_shapes(config):
    target = canonical_shapes(config)
    result = {native: target[canonical] for native, canonical in _DIRECT_MAP.items()}
    result.update({"decoder.layers." + native: (config.num_hidden_layers, *target["layers.0."+canonical])
                   for native, canonical in _LAYER_MAP.items()})
    if config.variant == "concat":
        result.update({native: target[canonical] for native, canonical in _FUSION_MAP.items()})
    return result


def tensor_dtype(name, config):
    return torch.float32 if name.endswith("expert_bias") else getattr(torch, config.dtype)


def convert_native_tensors(native, config):
    """Pure mapping/splitting: grouped QKV and expert GLU row order stay native."""
    expected = native_shapes(config)
    if set(native) != set(expected):
        raise ValueError(f"native key mismatch: missing={sorted(set(expected)-set(native))}, "
                         f"unexpected={sorted(set(native)-set(expected))}")
    for name, value in native.items():
        if tuple(value.shape) != expected[name] or value.dtype != tensor_dtype(name, config):
            raise ValueError(f"native shape/dtype mismatch: {name}: {tuple(value.shape)}, {value.dtype}")
    result = {canonical: native[name] for name, canonical in _DIRECT_MAP.items()}
    if config.variant == "concat":
        result.update({canonical: native[name] for name, canonical in _FUSION_MAP.items()})
    for name, canonical in _LAYER_MAP.items():
        for index in range(config.num_hidden_layers):
            result[f"layers.{index}.{canonical}"] = native["decoder.layers."+name][index].contiguous()
    return result


def inspect_checkpoint(checkpoint, config):
    """Read the trusted native checkpoint's small DCP metadata, never weights."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    with (checkpoint / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    expected = native_shapes(config)
    model_tensors = {name: value for name, value in metadata.state_dict_metadata.items()
                     if name.startswith(("decoder.", "embedding.", "output_layer.")) and hasattr(value, "size")}
    if set(model_tensors) != set(expected):
        raise ValueError(f"checkpoint tensor keys differ: missing={sorted(set(expected)-set(model_tensors))}; "
                         f"unexpected={sorted(set(model_tensors)-set(expected))}")
    for name, shape in expected.items():
        item = model_tensors[name]
        if tuple(item.size) != shape or item.properties.dtype != tensor_dtype(name, config):
            raise ValueError(f"checkpoint shape/dtype mismatch for {name}: {item}")
    return {"checkpoint": str(checkpoint), "native_tensor_count": len(expected),
            "canonical_tensor_count": len(canonical_shapes(config)),
            "model_bytes": sum(torch.Size(shape).numel() * torch.tensor([], dtype=tensor_dtype(name, config)).element_size()
                               for name, shape in expected.items()),
            "metadata_sha256": file_sha256(checkpoint / ".metadata")}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_proof(proof_path, checkpoint, config):
    proof_path = Path(proof_path).resolve(strict=True)
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("stage") != "formal":
        raise ValueError("export requires a completed formal checkpoint proof")
    if proof.get("iteration") != proof.get("horizon") or "100" not in proof.get("labels", []):
        raise ValueError("this converter requires the completed full-data checkpoint")
    checkpoint = Path(checkpoint).resolve(strict=True)
    if Path(proof["checkpoint"]).resolve(strict=True) != checkpoint:
        raise ValueError("proof binds a different checkpoint")
    model = proof["model"]
    contract = {
        "hidden_size": config.hidden_size, "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads, "num_query_groups": config.num_key_value_heads,
        "kv_channels": config.head_dim, "moe_latent_size": config.latent_size,
        "moe_ffn_hidden_size": config.intermediate_size, "num_moe_experts": config.num_experts,
        "moe_router_topk": config.num_experts_per_tok, "rotary_base": config.rope_theta,
        "vocabulary": config.vocab_size, "layernorm_epsilon": config.rms_norm_eps,
        "attention_output_gate": True, "qk_layernorm": True, "tied_embeddings": True,
        "parallel_streams": config.variant == "concat",
    }
    if config.variant == "concat":
        contract.update(prelude=config.prelude, core_a=config.core, core_b=config.core,
                        coda=config.coda, fusion="concat", fusion_residual=False)
    for key, value in contract.items():
        if model.get(key) != value:
            raise ValueError(f"proof model mismatch: {key}={model.get(key)!r}, expected {value!r}")
    inventory = proof["checkpoint_inventory"]
    actual_files = {item.name for item in checkpoint.iterdir() if item.is_file()}
    if actual_files != set(inventory):
        raise ValueError("checkpoint file inventory differs from its completion proof")
    for name, info in inventory.items():
        stat = (checkpoint / name).stat()
        if stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise ValueError(f"checkpoint changed after proof: {name}")
    for name, digest in proof["metadata_sha256"].items():
        if file_sha256(checkpoint / name) != digest:
            raise ValueError(f"checkpoint metadata hash mismatch: {name}")
    return {"proof_path": str(proof_path), "proof_sha256": file_sha256(proof_path),
            "iteration": proof["iteration"], "source_inventory_sha256": proof["source_inventory_sha256"],
            "checkpoint_inventory": inventory}


def _scratch_destination(path, scratch_root=None):
    """An export is always a new directory. A declared scratch root also confines it to that root."""
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError("export destination already exists; no overwrite or merge is supported")
    if scratch_root is not None:
        root = Path(scratch_root).resolve(strict=True)
        if not target.is_relative_to(root) or target == root:
            raise ValueError("model exports must be a new directory within approved scratch")
    return target


def export_checkpoint(checkpoint, proof, output, config, scratch_root=None):
    report = inspect_checkpoint(checkpoint, config)
    provenance = verify_proof(proof, checkpoint, config)
    target = _scratch_destination(output, scratch_root)
    # Retain input model buffers and output serialization margin; no optimizer is loaded.
    available = os.statvfs(next(parent for parent in (target, *target.parents) if parent.exists())
                           if scratch_root is None else Path(scratch_root))
    if available.f_bavail * available.f_frsize < report["model_bytes"] * 2 + 1024**3:
        raise RuntimeError("insufficient shared scratch free space for export plus serialization margin")
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from safetensors.torch import save_file
    native = {name: torch.empty(shape, dtype=tensor_dtype(name, config), device="cpu")
              for name, shape in native_shapes(config).items()}
    load(native, storage_reader=FileSystemReader(str(checkpoint)),
         planner=DefaultLoadPlanner(flatten_state_dict=False))
    converted = convert_native_tensors(native, config)
    if not all(bool(value.isfinite().all()) for value in converted.values()):
        raise ValueError("non-finite model tensor in checkpoint")
    # Recheck completion evidence after the potentially long read.
    verify_proof(proof, checkpoint, config)
    target.mkdir(parents=True, exist_ok=False)
    save_file(converted, str(target / "model.safetensors"), metadata={"format": "wide-serving-v1"})
    export_config = replace(config, branch_execution="serial", moe_backend="auto")
    (target / "config.json").write_text(json.dumps(export_config.to_dict(), indent=2) + "\n")
    manifest = dict(report, **provenance, format_version=1, optimizer_exported=False,
                    config_sha256=file_sha256(target / "config.json"),
                    weights_sha256=file_sha256(target / "model.safetensors"),
                    storage_note="Shared free space was checked; it is not a private quota reservation.")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@torch.no_grad()
def load_export(model, directory, verify_hashes=True):
    """Validate every canonical key, shape and dtype, then load this rank's slices."""
    from safetensors import safe_open
    # A failed reload must not leave an older cache eligible for inference.
    model.invalidate_router_caches()
    directory = Path(directory)
    disk_config = WideConfig.from_dict(json.loads((directory / "config.json").read_text()))
    # Execution choices may vary while architecture and all numerical constants cannot.
    cfg = replace(model.config, branch_execution=disk_config.branch_execution,
                  moe_backend=disk_config.moe_backend)
    if cfg != disk_config:
        raise ValueError("serving model architecture differs from export config")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported export manifest")
    if file_sha256(directory / "config.json") != manifest["config_sha256"]:
        raise ValueError("export config checksum mismatch")
    if verify_hashes and file_sha256(directory / "model.safetensors") != manifest["weights_sha256"]:
        raise ValueError("export weights checksum mismatch")
    expected = canonical_shapes(disk_config)
    destination = model.state_dict(keep_vars=True)
    if not set(destination) <= set(expected):
        raise ValueError("model contains tensors outside the canonical export schema")
    seen = set()
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as stream:
        if set(stream.keys()) != set(expected):
            raise ValueError("export has missing or unexpected tensor keys")
        for name, shape in expected.items():
            view = stream.get_slice(name)
            if tuple(view.get_shape()) != shape:
                raise ValueError(f"export tensor shape mismatch: {name}")
            dtype = "F32" if tensor_dtype(name, disk_config) == torch.float32 else "BF16"
            if view.get_dtype() != dtype:
                raise ValueError(f"export tensor dtype mismatch: {name}")
            if name in destination:
                value = model.shard_export_tensor(name, stream.get_tensor(name))
                dest = destination[name]
                if value.shape != dest.shape or value.dtype != dest.dtype:
                    raise ValueError(f"local tensor shape/dtype mismatch: {name}")
                dest.copy_(value)
                seen.add(name)
    if seen != set(destination):
        raise ValueError(f"not all local tensors were loaded: {set(destination)-seen}")
    model.prepare_router_caches()
    return {"canonical_tensors": len(expected), "loaded_local_tensors": len(seen),
            "weights_checksum_verified": verify_hashes,
            "router_backend": model.router_backend,
            "parallel_layout": model.parallel_layout()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=("standard", "concat"))
    parser.add_argument("--proof")
    parser.add_argument("--output")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--scratch-root", help="optional: confine the export to this root")
    args = parser.parse_args()
    config = WideConfig(variant=args.variant)
    if args.metadata_only:
        report = inspect_checkpoint(args.checkpoint, config)
        if args.proof:
            report["provenance"] = verify_proof(args.proof, args.checkpoint, config)
    else:
        if not args.output or not args.proof:
            parser.error("conversion requires --proof and --output")
        report = export_checkpoint(args.checkpoint, args.proof, args.output, config, args.scratch_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
