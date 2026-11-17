"""Opt-in same-weight native-reference parity suite; run on one allocated GPU.

The old implementation lives outside this repository. Each implementation
runs in a separate process to prevent reference monkeypatches leaking into
the explicit model. No legacy execution code is bundled with this test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def cases():
    outer = dict(scale="W1", hidden_size=256, latent_size=128, ffn_hidden_size=128)
    result = {"standard": {**outer, "family": "standard", "num_layers": 4}}
    result["independent-wide-asymmetric"] = {
        **outer, "family": "wide-shallow", "prelude": 1, "branches": [1, 3], "coda": 1}
    for k in (2, 3, 4):
        for mode in ("independent", "joint"):
            result[f"{mode}-narrow-{k}"] = {
                **outer, "family": "balanced", "prelude": 1, "branches": [2] * k,
                "coda": 1, "branch_hidden_size": 128, "branch_latent_size": 64,
                "branch_ffn_hidden_size": 64, "kv_mode": mode}
    return result


def metrics(reference, candidate):
    import torch
    if reference.shape != candidate.shape:
        raise AssertionError(f"Shape differs: {reference.shape} != {candidate.shape}")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise AssertionError("Nonfinite parity tensors")
    reference, candidate = reference.double(), candidate.double()
    difference = candidate - reference
    norm = float(torch.linalg.vector_norm(reference))
    difference_norm = float(torch.linalg.vector_norm(difference))
    return dict(max_abs=float(difference.abs().max()),
                relative_l2=difference_norm / norm if norm else (0.0 if difference_norm == 0 else float("inf")),
                reference_l2=norm)


def run_worker(command, environment, log, timeout):
    """Bound the worker and clean up its torchrun children on failure."""
    with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def compare(case_dir, model_config, *, gradient_rtol, output_rtol, output_atol):
    import torch
    from mbt.config import ModelConfig
    from mbt.model import convert_legacy_state_dict
    reference = torch.load(case_dir / "reference-result.pt", map_location="cpu", weights_only=False)
    candidate = torch.load(case_dir / "candidate-result.pt", map_location="cpu", weights_only=False)
    if reference["parameter_count"] != candidate["parameter_count"]:
        raise AssertionError("Trainable model parameter count changed")
    for key in ("logits", "causal_prefix"):
        torch.testing.assert_close(candidate[key], reference[key], rtol=output_rtol, atol=output_atol)
    if abs(candidate["loss"] - reference["loss"]) > 1e-4:
        raise AssertionError(f"CE changed: {reference['loss']} -> {candidate['loss']}")
    expected = convert_legacy_state_dict(reference["gradients"], ModelConfig.from_dict(model_config))
    actual = candidate["gradients"]
    if set(expected) != set(actual):
        raise AssertionError(f"Gradient keys differ; missing={sorted(set(expected)-set(actual))}, "
                             f"extra={sorted(set(actual)-set(expected))}")
    gradient_metrics = {}
    for name, ref in expected.items():
        got = actual[name]
        if (ref is None) != (got is None):
            raise AssertionError(f"Gradient participation changed: {name}")
        if ref is None:
            gradient_metrics[name] = {"unused": True}
            continue
        value = metrics(ref, got)
        gradient_metrics[name] = value
        # A relative norm criterion catches zeroed small gradients; a fixed
        # absolute tolerance alone can incorrectly accept entire missing paths.
        if value["relative_l2"] > gradient_rtol:
            raise AssertionError(f"Gradient mismatch: {name}: {value}")
    return dict(status="PASS", parameter_count=reference["parameter_count"],
                reference_loss=reference["loss"], candidate_loss=candidate["loss"],
                reference_causal_max_abs=reference["causal_max_abs"],
                candidate_causal_max_abs=candidate["causal_max_abs"],
                logits=metrics(reference["logits"], candidate["logits"]),
                causal_prefix=metrics(reference["causal_prefix"], candidate["causal_prefix"]),
                gradients=gradient_metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path,
                   default=os.environ.get("MBT_REFERENCE_ROOT"), required="MBT_REFERENCE_ROOT" not in os.environ)
    p.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--temp-root", type=Path, required=True)
    p.add_argument("--case", action="append", choices=tuple(cases()))
    p.add_argument("--gradient-rtol", type=float, default=0.01)
    p.add_argument("--output-rtol", type=float, default=0.002)
    p.add_argument("--output-atol", type=float, default=0.002)
    p.add_argument("--worker-timeout", type=int, default=300,
                   help="Maximum seconds for each isolated reference/candidate worker")
    args = p.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(Path.home().resolve()):
        raise ValueError("Parity artifacts must be on scratch, outside home")
    temporary = args.temp_root.resolve()
    if temporary.is_relative_to(Path.home().resolve()):
        raise ValueError("Temporary files must be outside home")
    if args.worker_timeout <= 0:
        raise ValueError("Worker timeout must be positive")
    temporary.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh parity output directory")
    output.mkdir(parents=True, exist_ok=True)
    candidate = args.candidate_root.resolve()
    reference = args.reference_root.resolve()
    if candidate == reference:
        raise ValueError("Reference and candidate must be distinct source trees")
    for source in (candidate, reference):
        if not (source / "code/mbt/config.py").is_file():
            raise ValueError(f"Missing MBT source tree: {source}")
    sys.path.insert(0, str(candidate / "code"))
    selected = args.case or list(cases())
    report = dict(reference_root=str(reference), candidate_root=str(candidate),
                  tolerances=dict(output_rtol=args.output_rtol, output_atol=args.output_atol,
                                  gradient_relative_l2=args.gradient_rtol, ce_absolute=1e-4), cases={})
    worker = Path(__file__).with_name("parity_worker.py")
    for name in selected:
        model = cases()[name]
        case = output / name
        case.mkdir()
        config = case / "model.json"
        config.write_text(json.dumps(model, indent=2) + "\n")
        for kind, source in (("reference", reference), ("candidate", candidate)):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(source / "code") + os.pathsep + str(source / "code/backend")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(temporary)
            for key in ("WANDB_SERVICE", "_WANDB_SERVICE"):
                env.pop(key, None)
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                   str(worker), "--implementation", kind, "--source-root", str(source),
                   "--config", str(config), "--case-dir", str(case), "--temp-root", str(args.temp_root)]
            with (case / f"{kind}.log").open("w") as log:
                run_worker(cmd, env, log, args.worker_timeout)
        report["cases"][name] = compare(case, model, gradient_rtol=args.gradient_rtol,
                                      output_rtol=args.output_rtol, output_atol=args.output_atol)
        (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": name, "status": "PASS", "logits": report["cases"][name]["logits"]}), flush=True)
    print(f"PARITY PASS: {len(selected)} cases; report: {output / 'parity.json'}", flush=True)


if __name__ == "__main__":
    main()
