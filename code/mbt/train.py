"""Portable torchrun entry point. Dry runs do not import PyTorch or touch GPUs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .training_config import DataConfig, Recipe, require, safe_name


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-config", type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Unique run directory on scratch/large storage; never the home filesystem")
    p.add_argument("--temp-root", type=Path, help="Short scratch path for per-process compiler/socket temporary files")
    p.add_argument("--backend-root", type=Path, help="Optional backend override; source and installed-wheel layouts are supported")
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--expert-parallel-size", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--keep-last-checkpoints", type=int)
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="Headroom beyond estimated checkpoint retention; does not check storage quotas")
    p.add_argument("--resume", action="store_true", help="Strictly resume this output directory's latest complete checkpoint")
    p.add_argument("--run-name", help="Optional short semantic name, also used by W&B when enabled")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-run-id")
    p.add_argument("--smoke-steps", type=int, help="Isolated mock-data run with a short schedule; not paper training")
    p.add_argument("--stop-after-step", type=int, help="Stop/save at this optimizer step without changing the LR horizon")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true", help="Required to start training; use torchrun")
    return p


def merge_args(*groups):
    """Reject conflicting duplicate native flags instead of silently overriding."""
    seen = set()
    result = []
    for group in groups:
        for item in group:
            if item.startswith("--"):
                require(item not in seen, f"Duplicate backend flag: {item}")
                seen.add(item)
            result.append(item)
    return result


def build_manifest(args):
    from .model import backend_path
    recipe = Recipe.load(args.config)
    updates = {}
    for cli, field in (("micro_batch_size", "micro_batch_size"), ("learning_rate", "learning_rate"),
                       ("keep_last_checkpoints", "keep_last_checkpoints")):
        if getattr(args, cli) is not None:
            updates[field] = getattr(args, cli)
    t = replace(recipe.training, **updates)
    if args.smoke_steps is not None:
        require(args.smoke_steps >= 2, "Smoke requires at least two updates")
        t = replace(t, train_steps=args.smoke_steps, sequence_length=128,
                    global_batch_size=args.world_size, micro_batch_size=1,
                    eval_global_batch_size=args.world_size, eval_micro_batch_size=1,
                    eval_steps=1, checkpoint_fraction=0.5)
    recipe = replace(recipe, training=t)
    require(args.expert_parallel_size in (1, 2, 4, 8), "Supported EP sizes: 1, 2, 4, 8")
    require(args.world_size % args.expert_parallel_size == 0, "World size must be divisible by EP")
    ga = t.accumulation_steps(args.world_size)
    require(args.min_free_gb >= 0, "Storage headroom must not be negative")
    if args.stop_after_step is not None:
        require(0 < args.stop_after_step < t.train_steps, "Stop step must precede the full schedule end")
    require(args.output_dir.is_absolute(), "Output path must be absolute")
    output = args.output_dir.resolve()
    require(output != Path('/'), "Output cannot be filesystem root")
    require(not output.is_relative_to(Path.home().resolve()), "Output must resolve outside the home directory")
    name = safe_name(args.run_name or (recipe.name + ("-smoke" if args.smoke_steps else "")))
    require(bool(args.wandb_entity) == bool(args.wandb_project), "W&B requires both explicit entity and project")
    require(not args.wandb_run_id or args.wandb_project, "W&B run ID requires a project")
    wandb = None
    if args.wandb_project:
        require(args.wandb_run_id is not None, "Give --wandb-run-id for deterministic W&B resume identity")
        wandb = dict(entity=args.wandb_entity, project=args.wandb_project,
                     name=name, run_id=safe_name(args.wandb_run_id))
    require(args.smoke_steps is not None or args.data_config is not None,
            "Formal training requires --data-config; mock data is restricted to --smoke-steps")
    data = DataConfig.load(args.data_config) if args.smoke_steps is None else None
    scalars = {
        "--train-iters": t.train_steps, "--global-batch-size": t.global_batch_size,
        "--micro-batch-size": t.micro_batch_size, "--seq-length": t.sequence_length,
        "--max-position-embeddings": t.sequence_length,
        "--eval-global-batch-size": t.eval_global_batch_size,
        "--eval-micro-batch-size": t.eval_micro_batch_size, "--eval-iters": t.eval_steps,
        "--eval-interval": t.checkpoint_interval, "--save-interval": t.checkpoint_interval,
        "--seed": t.seed, "--optimizer": "adam", "--lr": t.learning_rate,
        "--min-lr": t.min_lr_ratio * t.learning_rate, "--lr-decay-style": "cosine",
        "--lr-decay-iters": t.train_steps, "--lr-warmup-iters": max(1, int(t.train_steps * t.warmup_fraction)),
        "--lr-warmup-init": 0.0, "--weight-decay": t.weight_decay,
        "--adam-beta1": t.adam_beta1, "--adam-beta2": t.adam_beta2,
        "--adam-eps": t.adam_epsilon, "--clip-grad": t.clip_grad,
        "--expert-model-parallel-size": args.expert_parallel_size,
        "--recompute-granularity": "selective",
        "--num-workers": 0, "--dataloader-type": "single", "--log-interval": 1,
        "--save": output / "checkpoints", "--ckpt-format": "torch_dist",
        "--dist-ckpt-strictness": "raise_all", "--data-cache-path": output / "data-cache",
        "--tensorboard-dir": output / "tensorboard",
    }
    if args.resume:
        scalars["--load"] = output / "checkpoints"
    if args.stop_after_step is not None:
        scalars["--exit-interval"] = args.stop_after_step
    if wandb:
        scalars.update({"--wandb-entity": wandb["entity"], "--wandb-project": wandb["project"],
                        "--wandb-exp-name": name, "--wandb-save-dir": output / "wandb"})
    native = [v for key, value in scalars.items() for v in (key, str(value))]
    native += ["--log-validation-ppl-to-tensorboard"]
    if args.resume:
        native += ["--exit-on-missing-checkpoint"]
    native += data.backend_args() if data else ["--mock-data", "--tokenizer-type", "NullTokenizer"]
    native = merge_args(recipe.model.backend_args(), native)
    counts = recipe.model.counts(ep=args.expert_parallel_size)
    # Full BF16 model + FP32 master weights/Adam moments + metadata margin. A new
    # save must fit before the old save is removed. Quotas/concurrent writers are
    # site-specific and must still be checked by the person launching the job.
    checkpoint_bytes = int(counts["physical"] * 18)
    identity = dict(model_format="mbt-hierarchical-v2", recipe=recipe.to_dict(),
                    data=data.to_dict() if data else {"mock": True},
                    world_size=args.world_size, expert_parallel_size=args.expert_parallel_size, wandb=wandb)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(schema=2, name=name, identity_sha256=digest, **identity,
                output_dir=str(output), backend_root=str((args.backend_root or backend_path()).resolve()),
                native_arguments=native, gradient_accumulation_steps=ga,
                tokens=t.training_tokens, parameter_counts=counts,
                checkpoint_interval=t.checkpoint_interval,
                estimated_checkpoint_bytes=checkpoint_bytes,
                required_free_bytes=checkpoint_bytes * (t.keep_last_checkpoints + 1) + int(args.min_free_gb * 2**30),
                storage_check="Free-space estimate only; no quota or concurrent-writer guarantee",
                resume=args.resume, smoke=args.smoke_steps is not None,
                stop_after_step=args.stop_after_step)


def configure_environment(output, temp_root, rank):
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE"):
        os.environ.pop(key, None)
    temp_root = temp_root.resolve()
    require(temp_root.is_absolute() and not temp_root.is_relative_to(Path.home().resolve()),
            "Temporary files must resolve outside home")
    launch_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    suffix = hashlib.sha256((str(output) + launch_id).encode()).hexdigest()[:8]
    cache = output / ".cache" / suffix / f"rank{rank}"
    tmp = temp_root / f"mbt-{suffix}-{rank}"
    require(len(os.fsencode(tmp)) <= 70, "Temporary path too long; use a shorter --temp-root on scratch")
    directories = {"TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
                   "TORCHINDUCTOR_CACHE_DIR": cache / "inductor", "TRITON_CACHE_DIR": cache / "triton",
                   "CUDA_CACHE_PATH": cache / "cuda", "TORCH_EXTENSIONS_DIR": cache / "extensions",
                   "XDG_CACHE_HOME": cache / "xdg", "WANDB_DIR": output / "wandb",
                   "WANDB_CACHE_DIR": cache / "wandb", "WANDB_DATA_DIR": cache / "wandb-data"}
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    helper_cache = output / ".cache" / suffix / "dataset-helper"
    helper_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MBT_DATASET_HELPER_CACHE"] = str(helper_cache)
    # Preserve an explicitly selected HF cache; do not duplicate downloads.
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def prepare_run(args, manifest):
    import fcntl
    output = Path(manifest["output_dir"])
    rank = int(os.environ["RANK"])
    attempt = hashlib.sha256(os.environ.get("TORCHELASTIC_RUN_ID", "manual").encode()).hexdigest()[:16]
    ready = output / f".ready-{attempt}.json"
    lock = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        lock = (output / ".mbt.lock").open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output directory is already owned by another training process") from exc
        previous = output / "manifest.json"
        if args.resume:
            require(previous.is_file(), "Cannot resume without the original manifest")
            previous_manifest = json.loads(previous.read_text())
            require(previous_manifest.get("schema") == 2 and
                    previous_manifest.get("model_format") == "mbt-hierarchical-v2",
                    "Checkpoint uses an older model format; explicit MBT checkpoints are required")
            require(previous_manifest["identity_sha256"] == manifest["identity_sha256"],
                    "Resume changed model/training/data/layout/logger contract")
            checkpoint_root = output / "checkpoints"
            step = int((checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip())
            checkpoint = checkpoint_root / f"iter_{step:07d}"
            require((checkpoint / ".mbt-complete.json").is_file(), "Latest checkpoint lacks complete-save evidence")
            manifest["resume_step"] = step
            require(step < manifest["recipe"]["training"]["train_steps"], "Training already completed")
            require(args.stop_after_step is None or step < args.stop_after_step, "Stop step must follow resume step")
        else:
            require(not previous.exists(), "Run already exists; pass --resume or choose a fresh output directory")
            require(not (output / "checkpoints").exists(), "Fresh run cannot reuse a checkpoint directory")
            manifest["resume_step"] = 0
        require(shutil.disk_usage(output).free >= manifest["required_free_bytes"],
                "Insufficient free storage for checkpoint retention plus requested headroom")
        if not args.resume:
            previous.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary = ready.with_suffix(".tmp")
        temporary.write_text(json.dumps({"resume_step": manifest["resume_step"]}))
        temporary.replace(ready)
    else:
        deadline = time.monotonic() + 120
        while not ready.is_file():
            require(time.monotonic() < deadline, "Timed out waiting for rank-zero run admission")
            time.sleep(0.2)
        manifest["resume_step"] = json.loads(ready.read_text())["resume_step"]
    if args.resume:
        manifest["native_arguments"] += ["--ckpt-step", str(manifest["resume_step"])]
    temp_root = args.temp_root or Path(os.environ.get("MBT_TEMP_ROOT", str(output.parent / ".tmp")))
    configure_environment(output, temp_root, rank)
    if manifest["wandb"]:
        os.environ["WANDB_RUN_ID"] = manifest["wandb"]["run_id"]
        os.environ["WANDB_RESUME"] = "must" if args.resume else "never"
        os.environ["WANDB_MODE"] = "online"
        os.environ["MEGATRON_WANDB_FATAL"] = "1"
    return lock


def main(argv=None, *, lifecycle_factory=None):
    args = parser().parse_args(argv)
    require(not (args.execute and args.dry_run), "--execute and --dry-run are mutually exclusive")
    manifest = build_manifest(args)
    if args.dry_run or not args.execute:
        print(json.dumps(manifest, indent=2))
        return 0
    require("RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ,
            "Run training with torchrun (scripts/train.sh), including single-GPU training")
    require(int(os.environ["WORLD_SIZE"]) == args.world_size, "World-size argument differs from torchrun")
    backend = Path(manifest["backend_root"])
    require((backend / "megatron/training/training.py").is_file(), "Vendored training backend missing")
    if args.smoke_steps is None:
        DataConfig.load(args.data_config).validate_files()
    lock = prepare_run(args, manifest)
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from .pretrain import run
    try:
        run(manifest, lifecycle_factory=lifecycle_factory)
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
