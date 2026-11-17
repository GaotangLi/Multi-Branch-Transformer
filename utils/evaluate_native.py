"""Evaluate a native model bundle on an explicitly frozen token fixture.

Run with torchrun on one allocated CUDA GPU. The fixture contains tokens.npy
with shape [sequences, context + 1], an optional already-shifted binary loss
mask [sequences, context], and a checksum manifest. Nothing is downloaded,
tokenized, sampled, uploaded, or saved as a model checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time


TOKEN_DTYPES = {"int32", "int64"}
MASK_DTYPES = {"bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    """Validate the public fixture contract without importing NumPy or Torch."""
    require(manifest.get("schema_version") == 1, "Unsupported validation manifest version")
    context = manifest.get("context_length", 4096)
    count = manifest.get("num_sequences")
    require(type(context) is int and 1 <= context <= 4096, "context_length must be in 1..4096")
    require(type(count) is int and count > 0, "num_sequences must be a positive integer")
    require(isinstance(manifest.get("provenance"), dict), "Frozen-data provenance is required")
    expected = [("tokens", "tokens.npy", [count, context + 1], TOKEN_DTYPES)]
    if manifest.get("loss_mask") is not None:
        expected.append(("loss_mask", "loss_mask.npy", [count, context], MASK_DTYPES))
    for key, filename, shape, dtypes in expected:
        entry = manifest.get(key)
        require(isinstance(entry, dict), "Missing manifest entry: " + key)
        require(entry.get("file") == filename, "Fixture files must use the documented relative names")
        require(entry.get("shape") == shape, "Unexpected manifest shape: " + key)
        require(entry.get("dtype") in dtypes, "Unsupported fixture dtype: " + key)
        require(isinstance(entry.get("sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Missing or malformed checksum: " + key)
    return context, count


def sample_ranges(total, batch_size, max_samples=None):
    """Use an exact deterministic prefix, including a final partial batch."""
    require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    require(type(total) is int and total > 0, "No validation samples")
    if max_samples is not None:
        require(type(max_samples) is int and 1 <= max_samples <= total,
                "max_samples must be in 1..num_sequences")
    count = total if max_samples is None else max_samples
    return count, range(0, count, batch_size)


def split_batch(rows, mask, *, context, vocabulary):
    """Shift raw S+1-token windows once; masks already index the S targets."""
    import numpy as np
    require(rows.ndim == 2 and rows.shape[1] == context + 1, "Expected raw context+1 token windows")
    require(rows.dtype.name in TOKEN_DTYPES, "Token array must use int32/int64")
    require(bool((rows >= 0).all()) and bool((rows < vocabulary).all()), "Token ID outside the model vocabulary")
    inputs = np.array(rows[:, :-1], dtype=np.int64, order="C", copy=True)
    labels = np.array(rows[:, 1:], dtype=np.int64, order="C", copy=True)
    if mask is None:
        selected = np.ones(labels.shape, dtype=np.bool_)
    else:
        require(mask.shape == labels.shape and mask.dtype.name in MASK_DTYPES, "Loss-mask shape/dtype mismatch")
        require(bool(((mask == 0) | (mask == 1)).all()), "Loss masks must contain only exact 0/1 values")
        selected = np.array(mask, dtype=np.bool_, order="C", copy=True)
    return inputs, labels, selected


def summarize(loss_sum, token_count):
    require(type(token_count) is int and token_count > 0, "Validation contains no effective target tokens")
    require(math.isfinite(loss_sum) and loss_sum >= 0, "Invalid accumulated validation loss")
    loss = loss_sum / token_count
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    return {"loss_sum": loss_sum, "effective_token_count": token_count,
            "validation_loss": loss, "perplexity": perplexity,
            "perplexity_overflow": perplexity is None}


def outside_home(path, *, label):
    path = Path(path)
    require(path.is_absolute(), label + " must be an absolute scratch/large-storage path")
    resolved = path.resolve()
    require(resolved != Path("/") and not resolved.is_relative_to(Path.home().resolve()),
            label + " must resolve outside the home filesystem tree")
    return resolved


def configure_environment(output, temp_root):
    """Keep compiler caches and temporary files outside home before imports."""
    temp_root = outside_home(temp_root, label="Temporary root")
    identity = hashlib.sha256((str(output) + str(os.getpid())).encode()).hexdigest()[:10]
    temporary = temp_root / ("eval-" + identity)
    require(len(os.fsencode(temporary)) <= 70, "Use a shorter scratch --temp-root")
    cache = output.parent / ("." + output.stem + "-cache") / identity
    directories = {"TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor", "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda", "TORCH_EXTENSIONS_DIR": cache / "extensions",
        "XDG_CACHE_HOME": cache / "xdg"}
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE"):
        os.environ.pop(key, None)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    return {"cache": str(cache), "temporary": str(temporary)}


def file_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def load_fixture(directory):
    import numpy as np
    directory = Path(directory).resolve(strict=True)
    manifest_file = directory / "manifest.json"
    require(manifest_file.is_file() and manifest_file.stat().st_size <= 4 * 1024**2,
            "Missing or unreasonably large fixture manifest")
    manifest = json.loads(manifest_file.read_text())
    context, count = validate_manifest(manifest)
    identities = {manifest_file: file_identity(manifest_file)}
    arrays, file_hashes = {}, {}
    for name in ("tokens", "loss_mask"):
        entry = manifest.get(name)
        if entry is None:
            require(not (directory / "loss_mask.npy").exists(),
                    "A loss_mask.npy file exists but is not bound by the manifest")
            arrays[name] = None
            continue
        path = directory / entry["file"]
        require(path.is_file() and path.resolve().is_relative_to(directory), "Invalid fixture file path")
        identities[path] = file_identity(path)
        print("VALIDATION_VERIFY " + entry["file"], flush=True)
        digest = sha256(path)
        require(digest == entry["sha256"], "Validation checksum mismatch: " + entry["file"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(isinstance(array, np.memmap), "Validation arrays must be memory-mapped .npy files")
        require(list(array.shape) == entry["shape"] and array.dtype.name == entry["dtype"],
                "Validation array differs from its manifest")
        arrays[name], file_hashes[entry["file"]] = array, digest
    require(all(file_identity(path) == identity for path, identity in identities.items()),
            "Validation files changed while loading")
    return arrays["tokens"], arrays["loss_mask"], manifest, identities, {
        "manifest_sha256": sha256(manifest_file), "files": file_hashes,
        "context_length": context, "num_sequences": count,
        "provenance": manifest["provenance"], "tokenizer": manifest.get("tokenizer"),
    }


def frozen_state(model):
    """Record parameter versions and exact buffers without hashing model weights."""
    import torch
    parameters = {name: (id(value), value.data_ptr(), value._version, value.dtype, tuple(value.shape))
                  for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    routers = []
    for layer in model.decoder.layers:
        router = layer.mlp.router
        chunks = getattr(router, "qb_margin_chunks", None)
        require(chunks is None or not chunks, "Unexpected retained training router statistics")
        require(router.expert_bias.dtype == torch.float32, "Router bias must remain FP32")
        routers.append((router, None if chunks is None else len(chunks)))
    return parameters, buffers, routers


def check_frozen_state(model, before):
    import torch
    parameters, buffers, routers = before
    after = {name: (id(value), value.data_ptr(), value._version, value.dtype, tuple(value.shape))
             for name, value in model.named_parameters()}
    require(parameters == after, "Model parameters changed during evaluation")
    current = dict(model.named_buffers())
    require(set(current) == set(buffers), "Registered model buffers changed during evaluation")
    digest = hashlib.sha256()
    for name, original in buffers.items():
        value = current[name]
        require(value.shape == original.shape and value.dtype == original.dtype and torch.equal(value, original),
                "Model/router buffer changed: " + name)
        digest.update(name.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    for router, count in routers:
        chunks = getattr(router, "qb_margin_chunks", None)
        require((None if chunks is None else len(chunks)) == count, "Router statistics accumulated in eval")
    require(not any(module.training for module in model.modules()), "A module entered training mode")
    return {"parameter_versions_unchanged": True, "buffers_exactly_unchanged": True,
            "router_statistics_unchanged": True, "buffer_sha256": digest.hexdigest(),
            "parameters_checked": len(parameters), "buffers_checked": len(buffers)}


def atomic_result(path, result):
    temporary = path.with_name(path.name + ".partial-" + str(os.getpid()))
    require(not path.exists() and not temporary.exists(), "Refusing to overwrite evaluation results")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link publication fails rather than overwriting a concurrently created result.
    os.link(temporary, path)
    temporary.unlink()


def token_cross_entropy(logits, labels, chunk_size=256):
    """Return per-target FP32 CE without a full FP32 vocabulary-sized copy."""
    import torch
    from torch.nn import functional as F
    require(logits.ndim == 3 and labels.shape == logits.shape[:2], "Logits/labels shape mismatch")
    require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be positive")
    vocabulary = logits.shape[-1]
    flat_logits, flat_labels = logits.reshape(-1, vocabulary), labels.reshape(-1)
    losses = torch.empty(flat_labels.shape, device=logits.device, dtype=torch.float32)
    for offset in range(0, flat_labels.numel(), chunk_size):
        limit = min(offset + chunk_size, flat_labels.numel())
        losses[offset:limit] = F.cross_entropy(flat_logits[offset:limit].float(),
            flat_labels[offset:limit], reduction="none")
    return losses.view_as(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result on scratch/large storage")
    parser.add_argument("--max-samples", type=int, help="Explicit deterministic prefix smoke, not full validation")
    parser.add_argument("--per-sequence-losses", type=Path, help="Optional new .npy mean-loss array on scratch")
    parser.add_argument("--temp-root", type=Path, help="Short scratch root; defaults to MBT_TEMP_ROOT or TMPDIR")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    require(not socket.gethostname().split(".")[0].startswith("login"), "Run evaluation on a compute node")
    require(int(os.environ.get("WORLD_SIZE", "0")) == 1, "Use torchrun --standalone --nproc_per_node=1")
    require(args.progress_every > 0, "progress_every must be positive")
    output = outside_home(args.output, label="Output")
    require(not output.exists(), "Output already exists; choose a new result path")
    temp_root = args.temp_root or os.environ.get("MBT_TEMP_ROOT") or os.environ.get("TMPDIR")
    require(temp_root is not None, "Set a short scratch --temp-root, MBT_TEMP_ROOT, or TMPDIR")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = configure_environment(output, temp_root)
    sequence_path = None
    if args.per_sequence_losses:
        sequence_path = outside_home(args.per_sequence_losses, label="Per-sequence output")
        require(sequence_path != output and not sequence_path.exists(), "Per-sequence output must be new")
        require(sequence_path.suffix == ".npy", "Per-sequence output must end in .npy")
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
    # Source-checkout imports without hard-coded account or installation paths.
    repository = Path(__file__).resolve().parents[1]
    if (repository / "utils/native_checkpoint.py").is_file():
        sys.path.insert(0, str(repository))
    from utils import native_checkpoint
    native_checkpoint.import_public_model()
    from mbt.model import _prepare_backend
    _prepare_backend()
    import numpy as np
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    tokens, masks, data_manifest, data_identities, data_receipt = load_fixture(args.validation_data)
    context, total = validate_manifest(data_manifest)
    sample_count, starts = sample_ranges(total, args.batch_size, args.max_samples)
    bundle_manifest_file = args.bundle / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_file.read_text())
    bundle_config_file = args.bundle / "config.json"
    bundle_config = json.loads(bundle_config_file.read_text())
    vocabulary = bundle_config["architecture"]["vocabulary"]
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly one allocated CUDA GPU")
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=10))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, expert_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        print("VALIDATION_LOAD_MODEL", flush=True)
        model, model_receipt = native_checkpoint.load_model(args.bundle, max_sequence_length=context)
        model.eval().requires_grad_(False)
        require(not any(module.training for module in model.modules()), "Model must be entirely in eval mode")
        frozen = frozen_state(model)
        sequence_values = None if sequence_path is None else np.lib.format.open_memmap(
            sequence_path, mode="w+", dtype=np.float64, shape=(sample_count,))
        if sequence_values is not None:
            sequence_values[:] = np.nan
            sequence_values.flush()
        positions = torch.arange(context, device="cuda", dtype=torch.long)[None]
        batch_loss_sums, token_count, completed = [], 0, 0
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            for step, start in enumerate(starts, 1):
                end = min(start + args.batch_size, sample_count)
                batch = split_batch(tokens[start:end], None if masks is None else masks[start:end],
                                    context=context, vocabulary=vocabulary)
                inputs, labels, selected = [torch.from_numpy(value).to("cuda") for value in batch]
                logits = model(inputs, positions.expand(end - start, -1), None)
                require(tuple(logits.shape) == (end - start, context, vocabulary), "Unexpected logits shape")
                # Keep only one forward batch on device; cast CE chunks to FP32
                # instead of creating another full-vocabulary FP32 logits tensor.
                losses = token_cross_entropy(logits, labels)
                require(bool(torch.isfinite(losses).all()), "Nonfinite validation cross-entropy")
                sums = (losses.double() * selected).sum(-1).cpu().numpy()
                counts = selected.sum(-1).cpu().numpy()
                batch_loss_sums.append(float(sums.sum(dtype=np.float64)))
                token_count += int(counts.sum(dtype=np.int64))
                completed = end
                if sequence_values is not None:
                    sequence_values[start:end] = np.divide(sums, counts,
                        out=np.full(sums.shape, np.nan), where=counts > 0)
                if step % args.progress_every == 0 or end == sample_count:
                    current_sum = math.fsum(batch_loss_sums)
                    print(json.dumps({"sequences": completed, "total_sequences": sample_count,
                        "effective_tokens": token_count,
                        "running_loss": current_sum / token_count if token_count else None}), flush=True)
                del logits, losses, inputs, labels, selected
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        state_receipt = check_frozen_state(model, frozen)
        require(completed == sample_count, "Incomplete validation sample prefix")
        require(all(file_identity(path) == identity for path, identity in data_identities.items()),
                "Frozen validation files changed during evaluation")
        if sequence_values is not None:
            sequence_values.flush()
            del sequence_values
        metrics = summarize(math.fsum(batch_loss_sums), token_count)
        versions = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__}
        for distribution in ("transformer-engine", "safetensors"):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "unknown"
        result = {
            "schema_version": 1, "status": "COMPLETE",
            "scope": "prefix_smoke" if args.max_samples is not None else "full_validation",
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics, "sequences_evaluated": sample_count,
            "available_sequences": total, "sequence_prefix": [0, sample_count],
            "context_length": context, "batch_size": args.batch_size,
            "target_positions_evaluated": sample_count * context,
            "label_alignment": "input=tokens[:, :-1], label=tokens[:, 1:]; supplied loss mask is already target-aligned",
            "loss_definition": "sum(FP32 cross-entropy over binary-selected targets) / exact selected-token count; FP64 accumulation",
            "data": data_receipt,
            "model": {"bundle_name": args.bundle.resolve().name,
                "weights_sha256": bundle_manifest["weights_sha256"],
                "config_sha256": bundle_manifest["config_sha256"],
                "manifest_sha256": sha256(bundle_manifest_file), "load_receipt": model_receipt},
            "frozen_model": state_receipt,
            "execution": {"world_size": 1, "tensor_parallel": 1, "expert_parallel": 1,
                "precision": "bfloat16; router bias FP32", "gpu": torch.cuda.get_device_name(),
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "inference_seconds": inference_seconds, "total_seconds": time.perf_counter() - started,
                "effective_tokens_per_second": token_count / inference_seconds,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "versions": versions},
            "implementation": {"evaluator_sha256": sha256(__file__),
                               "native_loader_sha256": sha256(native_checkpoint.__file__)},
            "per_sequence_losses": None if sequence_path is None else {
                "file": sequence_path.name, "sha256": sha256(sequence_path),
                "shape": [sample_count], "dtype": "float64", "zero_target_sequences": "NaN"},
            "storage": environment,
        }
        atomic_result(output, result)
        print(json.dumps({"status": result["status"], "scope": result["scope"], **metrics,
                          "output": str(output)}), flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
