# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Write a small export with synthetic random weights, so the engine and the benchmark can be run
without a trained checkpoint.

The output is a normal `mbt-serving-v2` bundle (`config.json`, `manifest.json`,
`model.safetensors`) and is loaded by the same strict `load_export` path as a real export. Its
manifest records `trained: false`, so a run on it can never be mistaken for a measurement of a
trained model. Generated tokens are meaningless; throughput, memory and the exact serial/batched
parity check do not depend on the weight values.

Example: a Joint-KV four-branch model and the Standard model of the same depth:

    python tools/make_demo_export.py --output /tmp/demo-joint-k4 --variant mbt --kv-mode joint-branch-kv
    python tools/make_demo_export.py --output /tmp/demo-standard --variant standard
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.experimental.mbt_serving.config import MBTConfig
from nanovllm.experimental.mbt_serving.export import (canonical_shapes, expected_dtype, file_sha256,
                                                      scratch_destination)

# The fused router kernel the benchmark selects accepts exactly 64 experts, top-8 and renorm 2.5.
EXPERTS, TOPK, RENORM = 64, 8, 2.5


def build_config(args):
    layers = args.prelude + args.branches * args.branch_depth + args.coda
    shared = dict(hidden_size=args.hidden_size, num_attention_heads=args.hidden_size // args.head_dim,
                  num_key_value_heads=args.hidden_size // (2 * args.head_dim), head_dim=args.head_dim,
                  latent_size=args.latent_size, intermediate_size=args.intermediate_size,
                  num_experts=EXPERTS, num_experts_per_tok=TOPK, router_renorm_sum=RENORM,
                  vocab_size=args.vocab_size, max_position_embeddings=args.max_position_embeddings,
                  dtype=args.dtype)
    if args.variant == "standard":
        return MBTConfig(variant="standard", num_hidden_layers=args.layers or layers, **shared)
    return MBTConfig(variant="mbt", num_hidden_layers=layers, prelude=args.prelude, coda=args.coda,
                     branch_depths=(args.branch_depth,) * args.branches, input_bridge=True,
                     core_hidden_size=args.core_hidden_size, core_latent_size=args.core_latent_size,
                     core_intermediate_size=args.core_intermediate_size, kv_mode=args.kv_mode, **shared)


def random_weights(config, seed):
    """Match the initialization the tiny model tests use: norms near one, everything else small."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in canonical_shapes(config).items():
        dtype = getattr(torch, expected_dtype(name, config))
        value = torch.empty(shape, dtype=torch.float32)
        if name.endswith("expert_bias"):
            value = torch.linspace(-.2, .2, value.numel()).reshape(shape)
        elif "norm" in name:
            value.uniform_(.8, 1.2, generator=generator)
        else:
            value.normal_(0, .02, generator=generator)
        tensors[name] = value.to(dtype)
    return tensors


def write_export(config, output, seed, scratch_root=None):
    from safetensors.torch import save_file
    target = scratch_destination(output, scratch_root)
    tensors = random_weights(config, seed)
    target.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(target / "model.safetensors"), metadata={"format": "mbt-serving-v2"})
    export_config = replace(config, branch_execution="serial", moe_backend="auto")
    (target / "config.json").write_text(json.dumps(export_config.to_dict(), indent=2) + "\n")
    manifest = dict(format_version=2, bundle_format="mbt-serving-v2", trained=False,
                    weights="synthetic-random", seed=seed, optimizer_exported=False,
                    runtime_parity_qualified=False,
                    parameters=sum(value.numel() for value in tensors.values()),
                    model_bytes=sum(value.numel() * value.element_size() for value in tensors.values()),
                    provenance="tools/make_demo_export.py; no trained checkpoint; generated text is "
                               "meaningless and quality claims are not supported by this bundle",
                    config_sha256=file_sha256(target / "config.json"),
                    weights_sha256=file_sha256(target / "model.safetensors"))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new directory to write the bundle into")
    parser.add_argument("--variant", choices=("standard", "mbt"), default="mbt")
    parser.add_argument("--kv-mode", choices=("independent", "prelude-kv", "joint-branch-kv"),
                        default="joint-branch-kv", help="ignored for --variant standard")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-depth", type=int, default=2)
    parser.add_argument("--prelude", type=int, default=1)
    parser.add_argument("--coda", type=int, default=1)
    parser.add_argument("--layers", type=int, default=None,
                        help="standard only; defaults to the physical depth of the matching MBT model")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--core-hidden-size", type=int, default=256, help="branch width; a multiple of 2*head-dim")
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--core-latent-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--core-intermediate-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-position-embeddings", type=int, default=4096,
                        help="trained context; 4096 matches the published models")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch-root", default=None, help="optional: confine the output to this root")
    args = parser.parse_args()
    config = build_config(args)
    manifest = write_export(config, args.output, args.seed, args.scratch_root)
    print(json.dumps(dict(output=str(Path(args.output).resolve()), variant=config.variant,
                          kv_mode=config.kv_mode, branch_depths=list(config.branch_depths),
                          num_hidden_layers=config.num_hidden_layers, vocab_size=config.vocab_size,
                          parameters=manifest["parameters"], model_bytes=manifest["model_bytes"],
                          trained=False), indent=2))


if __name__ == "__main__":
    main()
