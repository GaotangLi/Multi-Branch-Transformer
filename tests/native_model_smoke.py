"""One-GPU native-model correctness checks, one configuration per process.

Run through torchrun after installing the GPU dependencies. No pretrained
weights or dataset are needed. This intentionally tests the production native
blocks at small widths, rather than an independently written toy Transformer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

CASES = ("standard", "independent-2", "independent-3", "independent-4",
         "joint-3", "joint-4", "balanced-joint-2", "balanced-independent-2",
         "asymmetric", "no-outer", "joint-no-outer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(Path.home().resolve()):
        raise ValueError("Use a scratch output directory, not home")
    output.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This mechanics test requests exactly one GPU")

    import torch
    from mbt import ModelConfig
    from mbt.model import build_model, make_native_config, backend_path

    options = dict(scale="W1", hidden_size=128, latent_size=64, ffn_hidden_size=64)
    if args.case == "standard":
        config = ModelConfig(**options, num_layers=4)
    elif args.case in ("balanced-joint-2", "balanced-independent-2"):
        options.update(hidden_size=256, latent_size=128, ffn_hidden_size=128)
        config = ModelConfig(**options, family="balanced", prelude=1, branches=(2,2), coda=1,
            branch_hidden_size=128, branch_latent_size=64, branch_ffn_hidden_size=80,
            kv_mode="joint" if args.case == "balanced-joint-2" else "independent")
    elif args.case == "asymmetric":
        config = ModelConfig(**options, family="wide-shallow", prelude=1, branches=(1,2,1), coda=1)
    elif args.case in ("no-outer", "joint-no-outer"):
        config = ModelConfig(**options, family="wide-shallow", prelude=0, branches=(1,1), coda=0,
            kv_mode="joint" if args.case == "joint-no-outer" else "independent")
    else:
        mode, count = args.case.rsplit("-", 1)
        config = ModelConfig(**options, family="wide-shallow", prelude=1,
            branches=(1,)*int(count), coda=1, kv_mode="joint" if mode == "joint" else "independent")

    sys.path.insert(0, str(backend_path()))
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, expert_model_parallel_size=1)
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    native = make_native_config(config, ep=1, gradient_accumulation_fusion=False)
    explicit_model = build_model(config, native_config=native, max_sequence_length=32).cuda()
    model = Float16Module(config=native, module=explicit_model)
    tokens = torch.randint(32, 4096, (1, 16), device="cuda")
    positions = torch.arange(16, device="cuda").unsqueeze(0)
    labels = torch.roll(tokens, -1, 1)
    model.train()
    loss = model(tokens, positions, None, labels=labels).float().mean()
    if not torch.isfinite(loss):
        raise AssertionError("non-finite native training loss")
    loss.backward()
    gradients = {name:p.grad for name,p in explicit_model.named_parameters() if p.grad is not None}
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients.values()):
        raise AssertionError("missing or non-finite native gradients")
    if config.branches:
        expected = "decoder.fusion.fusion_projection.weight"
        if expected not in gradients or gradients[expected].abs().max().item() == 0:
            raise AssertionError("fusion gradient is missing")
    if config.kv_mode == "joint":
        shared = [g for name,g in gradients.items() if "decoder.joint_kv." in name and "kv_projection" in name]
        if len(shared) != config.branches[0] or not all(g.abs().max().item() > 0 for g in shared):
            raise AssertionError("shared KV producers do not receive gradients")

    model.eval()
    changed = tokens.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 4096
    with torch.no_grad():
        reference = model(tokens, positions, None).detach()
        perturbed = model(changed, positions, None).detach()
    # Expert token-group sizes can change GEMM rounding even for unchanged
    # tokens. Use a BF16 tolerance rather than requiring different kernels to
    # be bitwise identical; save/restore below uses identical inputs and is exact.
    prefix_error = (reference[:, :8].float()-perturbed[:, :8].float()).abs().max().item()
    torch.testing.assert_close(reference[:, :8], perturbed[:, :8], rtol=1e-2, atol=2e-2)

    checkpoint = output / (args.case + ".pt")
    torch.save(explicit_model.state_dict(), checkpoint)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    explicit_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        restored = model(tokens, positions, None)
    torch.testing.assert_close(reference, restored, rtol=0, atol=0)
    checkpoint.unlink()
    result = dict(case=args.case, status="PASS", loss=loss.item(),
        parameter_count=sum(p.numel() for p in model.parameters()),
        gradient_tensors=len(gradients), causal_prefix_max_abs_error=prefix_error,
        causal_prefix_rtol=1e-2, causal_prefix_atol=2e-2,
        state_dict_roundtrip_exact=True, architecture=config.architecture())
    (output / (args.case + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
