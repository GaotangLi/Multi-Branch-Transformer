"""Explicit native-training callbacks for fixed validation and checkpoint retention.

Derived from the experiment lifecycle used for the paper. The native backend
still owns optimization, data sampling, distributed checkpointing and reload.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat

from .training_config import require

QB_FIELDS = ("expert_bias", "local_tokens_per_expert", "global_tokens_per_expert", "ga_steps", "qb_margin_chunks")


def clone(value):
    if hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone(v) for v in value]
    return deepcopy(value)


def equal(left, right, torch):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and bool(torch.equal(left, right))
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(equal(x, y, torch) for x, y in zip(left, right))
    return left == right


@contextmanager
def preserve_rng(torch):
    import random
    import numpy
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    py, np = random.getstate(), numpy.random.get_state()
    tracker = get_cuda_rng_tracker()
    states = deepcopy(tracker.get_states())
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(py)
        numpy.random.set_state(np)
        tracker.set_states(states)


def checkpoint_inventory(directory):
    """Stat metadata only; no expensive reread of tensor payloads."""
    require(directory.is_dir() and not directory.is_symlink(), "Checkpoint must be a real directory")
    required = ("metadata.json", ".metadata", "common.pt")
    require(all((directory / name).is_file() for name in required), "Checkpoint metadata incomplete")
    require(any(p.stat().st_size > 0 for p in directory.glob("*.distcp")), "Checkpoint shards are missing")
    result = {}
    for parent, subdirs, files in os.walk(directory, followlinks=False):
        for name in subdirs + files:
            path = Path(parent) / name
            require(not path.is_symlink(), "Checkpoint contains a symlink")
        for name in files:
            if name == ".mbt-complete.json":
                continue
            path = Path(parent) / name
            info = path.stat()
            require(stat.S_ISREG(info.st_mode), "Checkpoint contains a nonregular file")
            result[str(path.relative_to(directory))] = [info.st_size, info.st_mtime_ns]
    return result


def record_and_prune(checkpoints, step, identity, keep):
    """Prune only this run's unchanged, completed saves after the new save exists."""
    require(type(keep) is int and keep > 0, "Retention must keep at least one complete checkpoint")
    require(type(step) is int and step > 0, "Checkpoint step must be a positive integer")
    root = Path(checkpoints)
    require(root.resolve() == root and not root.is_symlink(), "Checkpoint root must be canonical")
    require((root / "latest_checkpointed_iteration.txt").read_text().strip() == str(step),
            "Latest-checkpoint tracker does not name the completed save")
    current = root / f"iter_{step:07d}"
    payload = dict(identity_sha256=identity, iteration=step, files=checkpoint_inventory(current))
    (current / ".mbt-complete.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    verified = []
    for child in root.iterdir():
        require(not child.is_symlink(), "Checkpoint root contains symlinks; refusing retention")
        if not re.fullmatch(r"iter_[0-9]{7,}", child.name):
            continue
        proof = child / ".mbt-complete.json"
        if not proof.is_file():
            continue  # Foreign/partial artifacts are never retention candidates.
        old = json.loads(proof.read_text())
        if old.get("identity_sha256") != identity:
            continue
        require(old["files"] == checkpoint_inventory(child), "Completed checkpoint changed after save")
        require(old["iteration"] <= step, "Refusing to prune a checkpoint newer than the current save")
        verified.append((old["iteration"], child, old))
    verified.sort()
    targets = verified[:-keep]
    require(getattr(shutil.rmtree, "avoids_symlink_attacks", False), "Safe retention needs fd-relative rmtree")
    removed = []
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        require(os.fstat(fd).st_ino == root.stat().st_ino, "Checkpoint root changed")
        for iteration, child, old in targets:
            require(old["files"] == checkpoint_inventory(child), "Checkpoint changed before deletion")
            require((root / "latest_checkpointed_iteration.txt").read_text().strip() == str(step),
                    "Another writer changed the checkpoint tracker")
            shutil.rmtree(child.name, dir_fd=fd)
            removed.append(iteration)
    finally:
        os.close(fd)
    return dict(current=step, removed=removed, kept=[i for i, _, _ in verified[-keep:]])


class TrainingLifecycle:
    """Per-run state, passed explicitly to the native pretraining loop.

    No module functions or model forwards are replaced. The backend invokes the
    methods at model construction, loader construction, evaluation and saving.
    """
    def __init__(self, manifest, training, rerun_iterator):
        self.manifest = manifest
        self.training = training
        self.rerun_iterator = rerun_iterator
        self.fixed_loader = None
        self.last_losses = None
        self.validation_active = False

    def model_ready(self, model, optimizer, config, opt_param_scheduler=None):
        """Extension point for bounded model checks before native training."""

    def before_checkpoint(self, iteration, non_persistent=False):
        """Extension point before the native complete-state save begins."""

    def data_loaders_ready(self, result):
        require(len(result[1]) == 1 and result[1][0] is not None,
                "Exactly one validation blend is required")
        # On resume, the native loader may start partway through validation.
        # Rebuild only its sampler, retaining the exact existing dataset object.
        self.fixed_loader = self.training.build_pretraining_data_loader(result[1][0].dataset, 0)
        return result

    def before_train(self, model, optimizer, config):
        native = self.training.get_args()
        start = self.manifest["resume_step"]
        require(native.iteration == start and native.consumed_train_samples == start * native.global_batch_size,
                "Native resume did not restore the optimizer-step/data cursor")
        require(self.fixed_loader is not None, "Fixed validation loader missing")
        require(not native.async_save, "Retention requires synchronous complete saves")
        for flag in ("no_save_optim", "no_save_rng", "no_load_optim", "no_load_rng", "finetune"):
            require(not getattr(native, flag, False), f"Full-state resume requires {flag}=False")
        writer = self.training.get_wandb_writer()
        if self.manifest["wandb"] and self.training.torch.distributed.get_rank() == native.world_size - 1:
            require(writer is not None, "Required online W&B writer is missing")
        if writer:
            writer.config.update({"mbt": self.manifest["recipe"],
                                  "identity_sha256": self.manifest["identity_sha256"]})
        print("MBT_TRAIN_START " + json.dumps({"step": start,
              "consumed_train_samples": native.consumed_train_samples}), flush=True)

    def on_validation_result(self, prefix, losses, incomplete):
        if "on test set" in prefix:
            return
        require(self.validation_active, "Validation result outside its lifecycle context")
        require(not incomplete and losses is not None, "Validation did not finish")
        values = {key: float(value.item() if hasattr(value, "item") else value)
                  for key, value in losses.items()}
        require("lm loss" in values and all(math.isfinite(v) for v in values.values()),
                "Invalid validation losses")
        self.last_losses = values

    @contextmanager
    def validation_context(self, prefix, model, config, iteration, data_iterator):
        if "on test set" in prefix:
            yield data_iterator
            return
        require(self.fixed_loader is not None, "Fixed validation prefix is unavailable")
        require(not self.validation_active, "Nested validation is unsupported")
        a, torch = self.training.get_args(), self.training.torch
        modules = list({id(m): m for chunk in model for m in chunk.modules()}.values())
        modes = [(m, m.training) for m in modules]
        qb = [(m, key, clone(getattr(m, key))) for m in modules for key in QB_FIELDS
              if getattr(m, key, None) is not None]
        cursors = {key: getattr(a, key) for key in
                   ("consumed_train_samples", "consumed_valid_samples", "skipped_train_samples")
                   if hasattr(a, key)}
        timers = getattr(config, "timers", None)
        rerun = self.training.get_rerun_state_machine()
        rerun_mode = rerun.get_mode()
        completed = False
        self.last_losses = None
        self.validation_active = True
        try:
            with preserve_rng(torch):
                a.consumed_valid_samples = 0
                yield self.rerun_iterator(iter(self.fixed_loader))
                require(a.consumed_valid_samples == a.eval_iters * a.eval_global_batch_size,
                        "Validation did not consume the fixed sample budget")
                require(self.last_losses is not None, "Validation produced no loss record")
                completed = True
        finally:
            unchanged = all(equal(getattr(m, key), saved, torch) for m, key, saved in qb)
            train_unchanged = all(getattr(a, key) == val for key, val in cursors.items()
                                  if key != "consumed_valid_samples")
            with torch.no_grad():
                for m, key, saved in qb:
                    current = getattr(m, key)
                    if torch.is_tensor(current): current.copy_(saved)
                    elif isinstance(current, list): current[:] = saved
                    else: setattr(m, key, saved)
            for m, mode in modes: m.training = mode
            for key, value in cursors.items(): setattr(a, key, value)
            if config is not None: config.timers = timers
            rerun.set_mode(rerun_mode)
            self.validation_active = False
            if completed:
                require(unchanged and train_unchanged,
                        "Validation altered router state or train cursor; restored and stopped")
                record = dict(step=iteration, losses=self.last_losses,
                              validation_start_sample=0,
                              validation_samples=a.eval_iters*a.eval_global_batch_size)
                if torch.distributed.get_rank() == 0:
                    with (Path(self.manifest["output_dir"]) / "validation.jsonl").open("a") as out:
                        out.write(json.dumps(record) + "\n")
                    print("MBT_VALIDATION " + json.dumps(record), flush=True)

    def after_checkpoint(self, iteration, non_persistent=False):
        require(not non_persistent, "Unexpected alternate checkpoint lifecycle")
        require(not self.training.get_args().async_save, "Complete-save callbacks require synchronous saving")
        torch = self.training.torch
        torch.distributed.barrier()
        status = [None]
        if torch.distributed.get_rank() == 0:
            try:
                report = record_and_prune(Path(self.manifest["output_dir"]) / "checkpoints", iteration,
                                          self.manifest["identity_sha256"],
                                          self.manifest["recipe"]["training"]["keep_last_checkpoints"])
                status[0] = {"ok": True, **report}
                with (Path(self.manifest["output_dir"]) / "checkpoint-history.jsonl").open("a") as out:
                    out.write(json.dumps(status[0]) + "\n")
                print("MBT_CHECKPOINT " + json.dumps(status[0]), flush=True)
            except Exception as error:
                status[0] = {"ok": False, "error": str(error)}
        torch.distributed.broadcast_object_list(status, src=0)
        require(status[0]["ok"], f"Complete-save/retention failed: {status[0]}")
