"""One isolated worker for the opt-in native-reference parity check.

Use compare_reference.py on an allocated GPU. The reference implementation is
provided externally; it is never imported into the candidate worker process.
All artifacts and caches belong in the explicitly supplied scratch directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--implementation", choices=("reference", "candidate"), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--temp-root", type=Path, required=True)
    args = p.parse_args()
    source = args.source_root.resolve()
    case = args.case_dir.resolve()
    temporary = args.temp_root.resolve()
    for path in (case, temporary):
        if path.is_relative_to(Path.home().resolve()):
            raise ValueError("Parity artifacts and temporary files must be outside home")
    case.mkdir(parents=True, exist_ok=True)
    cache = case.parent / ".cache" / args.implementation
    suffix = hashlib.sha256(str(case).encode()).hexdigest()[:8]
    temp = temporary / f"mbtp-{suffix}-{args.implementation[0]}"
    if len(os.fsencode(temp)) > 70:
        raise ValueError("Use a shorter scratch --temp-root")
    paths = {
        "TMPDIR": temp, "TMP": temp, "TEMP": temp,
        "TORCH_EXTENSIONS_DIR": cache / "extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
        "CUDA_CACHE_PATH": cache / "cuda",
        "XDG_CACHE_HOME": cache / "xdg",
        "MBT_DATASET_HELPER_CACHE": cache / "dataset-helper",
    }
    for key, directory in paths.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE"):
        os.environ.pop(key, None)
    backend = source / "code" / "backend"
    sys.path[:0] = [str(source / "code"), str(backend)]

    import torch
    from mbt.config import ModelConfig

    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    if args.implementation == "reference":
        from mbt.model import install
        install(backend, config)

    # Both workers construct the same native config, process groups, kernels,
    # RNG trackers and dtype wrappers. No training data or tokenizer is needed.
    sys.argv = ["mbt-parity", *config.backend_args(),
                "--seq-length", "16", "--max-position-embeddings", "16",
                "--micro-batch-size", "1", "--global-batch-size", "1",
                "--train-iters", "4", "--lr", "0.0001",
                "--lr-decay-iters", "4", "--seed", "1234",
                "--tokenizer-type", "NullTokenizer", "--vocab-size", "128256",
                "--mock-data", "--num-workers", "0",
                "--no-gradient-accumulation-fusion"]
    from megatron.training.arguments import parse_and_validate_args, core_transformer_config_from_args
    from megatron.training.initialize import initialize_megatron
    from megatron.core.transformer.module import Float16Module

    native_args = parse_and_validate_args()
    initialize_megatron()
    native_config = core_transformer_config_from_args(native_args)
    native_config.gradient_accumulation_fusion = False
    if args.implementation == "reference":
        from gpt_builders import gpt_builder
        raw = gpt_builder(native_args, True, True, config=native_config)
    else:
        from mbt.model import build_model, convert_legacy_state_dict
        raw = build_model(config, native_config=native_config,
                          max_sequence_length=16, vocab_size=128256,
                          backend_root=backend)
    raw = raw.cuda()
    model = Float16Module(config=native_config, module=raw)
    weights_path = case / "reference-weights.pt"
    if args.implementation == "reference":
        torch.save(raw.state_dict(), weights_path)
    else:
        # This is an explicitly supplied, locally generated test artifact.
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = convert_legacy_state_dict(state, config)
        raw.load_state_dict(state, strict=True)
        loaded = raw.state_dict()
        for name, expected in state.items():
            if torch.is_tensor(expected):
                torch.testing.assert_close(loaded[name].detach().cpu(), expected.cpu(), rtol=0, atol=0)
        del state

    tokens = ((torch.arange(32, device="cuda").view(2, 16) * 17 + 11) % 997).long()
    positions = torch.arange(16, device="cuda").expand(2, -1)
    labels = (tokens + 17) % 997
    model.eval()
    with torch.no_grad():
        logits_eval = model(tokens, positions, None)
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 313) % 997
        changed_logits = model(changed, positions, None)
        # A suffix changes expert-group GEMM shapes; BF16 rounding can therefore
        # change prefix values slightly even with strictly causal attention.
        # Record the error rather than claiming bitwise prefix equality.
        causal_max_abs = float((logits_eval[:, :8] - changed_logits[:, :8]).abs().max())
        torch.testing.assert_close(logits_eval[:, :8], changed_logits[:, :8], rtol=1e-2, atol=2e-2)
        if torch.equal(logits_eval[:, 8:], changed_logits[:, 8:]):
            raise AssertionError("Model ignored changed suffix tokens")
        prefix = logits_eval[:, :8].float().cpu()
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(tokens, positions, None)
    if tuple(logits.shape) != (2, 16, 128256):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 128256), labels.reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise AssertionError("Nonfinite parity objective")
    loss.backward()
    gradients = {}
    for name, parameter in raw.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"Nonfinite gradient: {name}")
        gradients[name] = None if gradient is None else gradient.detach().float().cpu()
    result = dict(config=config.to_dict(), implementation=args.implementation,
                  logits=logits.detach().float().cpu(), loss=float(loss.detach()),
                  causal_prefix=prefix, gradients=gradients,
                  parameter_count=sum(p.numel() for p in raw.parameters()),
                  causality="PASS", causal_max_abs=causal_max_abs,
                  causal_tolerance=dict(rtol=1e-2, atol=2e-2), torch_version=torch.__version__)
    torch.save(result, case / f"{args.implementation}-result.pt")
    print("MBT_PARITY_WORKER " + json.dumps({
        "implementation": args.implementation, "loss": result["loss"],
        "parameters": result["parameter_count"], "causality": "PASS",
        "causal_max_abs": causal_max_abs}), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
