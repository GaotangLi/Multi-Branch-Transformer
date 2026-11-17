"""Prepare the frozen MBT validation tokens for utils/evaluate_native.py.

Download the public Parquet dataset, or use --parquet for an existing local
copy. Rows are copied in their original order, without tokenization, shifting,
padding, or sampling. Downloads and output belong on scratch/large storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


DATASET_REPO = "gaotang/MBT-Validation"
# Immutable revision of the fixed validation release.
DEFAULT_REVISION = "bd708088668e16c74a422d2acbae7a911aa158d2"
PARQUET_FILE = "data/validation-00000-of-00001.parquet"
EXPECTED_SHAPE = (12160, 4097)
VOCABULARY_SIZE = 128256
TOKENS_SHA256 = "03475fe10c3309ef89dc679f92baa772f8f742fb010584657cb192f1f83fc8b2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _convert_parquet(parquet, output, *, expected_shape=EXPECTED_SHAPE,
                     expected_sha256=TOKENS_SHA256, vocabulary_size=VOCABULARY_SIZE,
                     batch_size=128, provenance=None):
    """Bounded-memory conversion; internal overrides support tiny offline tests.

    The CLI deliberately exposes no shape/hash override: public evaluation
    requires the exact released fixture. A manifest is written only after all
    checks pass; an interrupted or rejected conversion has no valid manifest.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    require(len(expected_shape) == 2 and all(type(x) is int and x > 0 for x in expected_shape),
            "Expected shape must contain two positive integer dimensions")
    require(expected_shape[1] >= 2, "Each row must contain input tokens and a next-token target")
    require(type(vocabulary_size) is int and vocabulary_size > 0, "Invalid vocabulary size")
    require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "An exact expected token-file SHA256 is required")
    parquet = Path(parquet).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    require(parquet.is_file(), "Parquet input must be a file")
    require(not output.exists(), "Output already exists; select a NEW directory")
    original_identity = file_identity(parquet)
    source = pq.ParquetFile(parquet)
    schema = source.schema_arrow
    require(schema.names == ["input_ids"], "Expected exactly one column: input_ids")
    field_type = schema.field("input_ids").type
    require(pa.types.is_list(field_type) and pa.types.is_int32(field_type.value_type),
            "input_ids must have Arrow type list<int32>")
    rows, width = expected_shape
    require(source.metadata.num_rows == rows, "Parquet row count differs from the released fixture")
    # mkdir(exist_ok=False) reserves this destination and never overwrites a
    # pre-existing directory, including one created concurrently.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    partial = output / ".tokens.npy.partial"
    tokens = np.lib.format.open_memmap(partial, mode="w+", dtype="<i4",
                                      shape=expected_shape, fortran_order=False, version=(1, 0))
    offset = 0
    try:
        for batch in source.iter_batches(batch_size=batch_size, columns=["input_ids"], use_threads=False):
            column = batch.column(0)
            require(column.null_count == 0, "Null token rows are not permitted")
            lengths = np.diff(column.offsets.to_numpy(zero_copy_only=False))
            require(bool((lengths == width).all()), "Every row must contain exactly the released number of tokens")
            flattened = column.flatten()
            require(flattened.null_count == 0, "Null token IDs are not permitted")
            values = flattened.to_numpy(zero_copy_only=False)
            require(values.size == batch.num_rows * width, "Unexpected flattened token count")
            require(bool((values >= 0).all()) and bool((values < vocabulary_size).all()),
                    "Token ID is outside the released vocabulary")
            end = offset + batch.num_rows
            require(end <= rows, "Too many Parquet rows")
            tokens[offset:end] = values.reshape(batch.num_rows, width)
            offset = end
        require(offset == rows, "Incomplete Parquet token sequence")
        tokens.flush()
    finally:
        del tokens
        source.close()
    # Force completed bytes to storage before binding the final manifest.
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    digest = sha256(partial)
    require(digest == expected_sha256,
            "Token-file SHA256 mismatch: order/content/format differs from the released fixture")
    source_digest = sha256(parquet)
    require(file_identity(parquet) == original_identity, "Parquet input changed during conversion")
    os.rename(partial, output / "tokens.npy")
    details = dict(provenance or {})
    details.update({"parquet_file": parquet.name, "parquet_sha256": source_digest,
                    "row_order": "original Parquet row order; no shuffling or repacking",
                    "loss_mask": "all targets included; no mask file required"})
    manifest = {
        "schema_version": 1, "context_length": width - 1, "num_sequences": rows,
        "tokens": {"file": "tokens.npy", "sha256": digest,
                   "shape": list(expected_shape), "dtype": "int32"},
        "provenance": details,
    }
    pending = output / ".manifest.json.partial"
    with pending.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New validation directory on scratch/large storage")
    parser.add_argument("--parquet", type=Path, help="Use this local Parquet file instead of downloading")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face dataset revision (release commit recommended)")
    parser.add_argument("--cache-dir", type=Path, help="Explicit large-storage HF cache; REQUIRED when downloading")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Output already exists; select a NEW directory")
    if args.parquet is not None:
        parquet = args.parquet
        provenance = {"source": "local Parquet"}
    else:
        require(args.cache_dir is not None, "Provide --cache-dir on scratch/large storage for dataset downloads")
        require(args.revision, "Provide --revision until an immutable release default is configured")
        from huggingface_hub import snapshot_download
        cache = args.cache_dir.expanduser().resolve()
        snapshot = Path(snapshot_download(repo_id=args.repo_id, repo_type="dataset",
            revision=args.revision, cache_dir=str(cache), allow_patterns=[PARQUET_FILE], max_workers=1))
        parquet = snapshot / PARQUET_FILE
        require(parquet.is_file(), "The requested dataset revision has no release Parquet file")
        provenance = {"source": "Hugging Face dataset", "dataset_repo": args.repo_id,
                      "requested_revision": args.revision, "snapshot_revision": snapshot.name}
    manifest = _convert_parquet(parquet, output, provenance=provenance)
    print(json.dumps({"status": "READY", "output": str(output),
        "num_sequences": manifest["num_sequences"], "context_length": manifest["context_length"],
        "tokens_sha256": manifest["tokens"]["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
