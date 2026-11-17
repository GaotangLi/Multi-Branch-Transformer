"""Convert the pinned FineWeb-Edu sample-100BT plan to Megatron indexed data.

Run on allocated CPU resources in the project training environment. Inputs are
already-downloaded Parquet files and Meta's original Llama-3 tokenizer.model.
The destination must be new. No downloads, resume, or training are performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

VOCAB_SIZE = 128256
EOD_ID = 128001
TRAIN_SHARDS = 64
SOURCE_FILES = 140
REPOSITORY = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".partial")
    require(not path.exists() and not temporary.exists(), "Refusing to overwrite " + str(path))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def validate_plan(plan):
    require(plan.get("schema_version") == 1, "Unsupported data-plan version")
    for field in ("dataset_repo", "dataset_revision", "tokenizer_sha256"):
        require(isinstance(plan.get(field), str) and plan[field], "Missing plan field: " + field)
    require(re.fullmatch(r"[0-9a-f]{64}", plan["tokenizer_sha256"]), "Invalid tokenizer checksum")
    groups = plan.get("training_sources")
    require(isinstance(groups, list) and len(groups) == TRAIN_SHARDS, "Expected 64 ordered training groups")
    require(all(isinstance(group, list) and group for group in groups), "Training groups must be nonempty lists")
    names = [name for group in groups for name in group] + [plan.get("validation_source")]
    require(all(isinstance(name, str) and Path(name).name == name and name.endswith(".parquet")
                for name in names), "Plan sources must be Parquet basenames")
    require(len(names) == SOURCE_FILES and len(set(names)) == SOURCE_FILES,
            "Plan must contain 140 unique sources, with validation excluded from training")
    return groups, plan["validation_source"]


def source_paths(raw_dir, plan):
    groups, validation = validate_plan(plan)
    expected = {name for group in groups for name in group} | {validation}
    actual = {path.name for path in raw_dir.glob("*.parquet") if path.is_file()}
    require(actual == expected, "Raw Parquet files differ from the plan; missing="
            + repr(sorted(expected - actual)) + "; extra=" + repr(sorted(actual - expected)))
    return [[raw_dir / name for name in group] for group in groups], [raw_dir / validation]


def encode_text_batch(encoding, texts):
    """Original encoding policy: ordinary text, 400k-character chunks, no BOS."""
    limit = 400_000
    outputs = [None] * len(texts)
    short_indices = [index for index, text in enumerate(texts) if len(text) <= limit]
    if short_indices:
        encoded = encoding.encode_ordinary_batch(
            [texts[index] for index in short_indices], num_threads=1)
        for index, token_ids in zip(short_indices, encoded, strict=True):
            outputs[index] = token_ids
    for index, text in enumerate(texts):
        if outputs[index] is None:
            outputs[index] = sum((encoding.encode_ordinary(text[start:start + limit])
                                 for start in range(0, len(text), limit)), start=[])
    return [token_ids for token_ids in outputs if token_ids is not None]


@lru_cache(maxsize=1)
def _encoding(tokenizer_model):
    backend = REPOSITORY / "code/backend"
    require(backend.is_dir(), "Run this utility from the public repository's utils directory")
    sys.path.insert(0, str(backend))
    import torch
    import pyarrow as pa
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    from megatron.core.tokenizers.text.libraries.tiktoken_tokenizer import TikTokenTokenizer
    tokenizer = TikTokenTokenizer(tokenizer_model, num_special_tokens=256, vocab_size=VOCAB_SIZE)
    require(tokenizer.eod == EOD_ID and tokenizer.vocab_size == VOCAB_SIZE,
            "Unexpected tokenizer vocabulary/EOD layout")
    return tokenizer.tokenizer


def process_shard(task):
    """Convert an ordered source group and verify its native index before commit."""
    encoding = _encoding(task["tokenizer_model"])
    import numpy as np
    import pyarrow.parquet as pq
    from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
    prefix = Path(task["output_prefix"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {suffix: Path(str(prefix) + suffix) for suffix in
                 (".bin", ".idx", ".partial.bin", ".partial.idx", ".stats.json")}
    require(not any(path.exists() for path in artifacts.values()), "Shard output already exists: " + prefix.name)
    builder = IndexedDatasetBuilder(str(artifacts[".partial.bin"]), dtype=np.int32)
    documents = tokens = input_bytes = 0
    identities = {}
    started = time.monotonic()
    try:
        for filename in task["sources"]:
            source = Path(filename)
            stat = source.stat()
            identities[source] = (stat.st_size, stat.st_mtime_ns)
            input_bytes += stat.st_size
            with pq.ParquetFile(source) as parquet:
                for batch in parquet.iter_batches(batch_size=128, columns=["text"], use_threads=False):
                    texts = batch.column(0).to_pylist()
                    require(all(isinstance(text, str) for text in texts), "Non-string text in " + source.name)
                    for token_ids in encode_text_batch(encoding, texts):
                        # Empty documents intentionally become one EOD token.
                        token_ids.append(EOD_ID)
                        builder.add_document(token_ids, [len(token_ids)])
                        documents += 1
                        tokens += len(token_ids)
                    if documents and documents % 100000 < len(texts):
                        print(json.dumps({"shard": prefix.name, "documents": documents, "tokens": tokens}), flush=True)
        require(documents > 0, "An output shard has no documents")
        builder.finalize(str(artifacts[".partial.idx"]))
    except BaseException:
        builder.data_file.close()
        raise
    require(all((path.stat().st_size, path.stat().st_mtime_ns) == identity
                for path, identity in identities.items()), "A source changed during preprocessing")
    partial_prefix = str(prefix) + ".partial"
    dataset = IndexedDataset(partial_prefix, multimodal=False, mmap=True, object_storage_config=None)
    require(dataset.index.dtype == np.int32, "Unexpected indexed token dtype")
    require(dataset.index.document_count - 1 == documents and len(dataset) == documents,
            "Indexed document count mismatch")
    require(bool((dataset.sequence_lengths > 0).all())
            and int(np.sum(dataset.sequence_lengths, dtype=np.int64)) == tokens,
            "Indexed token count mismatch")
    require(artifacts[".partial.bin"].stat().st_size == tokens * 4, "Incomplete token binary")
    del dataset
    for suffix in (".bin", ".idx"):
        partial = artifacts[".partial" + suffix]
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.rename(partial, artifacts[suffix])
    record = {"status": "complete", "prefix": task["relative_prefix"],
        "sources": [Path(path).name for path in task["sources"]],
        "documents": documents, "tokens": tokens, "input_bytes": input_bytes,
        "bin_bytes": artifacts[".bin"].stat().st_size, "idx_bytes": artifacts[".idx"].stat().st_size,
        "elapsed_seconds": time.monotonic() - started}
    atomic_json(artifacts[".stats.json"], record)
    print(json.dumps({"shard": prefix.name, "status": "complete", "documents": documents, "tokens": tokens}), flush=True)
    return record


def prepare(raw_dir, tokenizer_model, output_dir, plan_path, workers=8):
    require(type(workers) is int and workers > 0, "workers must be positive")
    raw_dir, tokenizer_model, plan_path = (Path(path).expanduser().resolve(strict=True)
                                          for path in (raw_dir, tokenizer_model, plan_path))
    output_dir = Path(output_dir).expanduser().resolve()
    require(not output_dir.exists() and output_dir != Path(output_dir.anchor),
            "Choose a NEW output directory; overwrite/resume is not supported")
    require(tokenizer_model.is_file() and tokenizer_model.suffix == ".model",
            "Use Meta's original tokenizer.model file")
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    groups, validation = source_paths(raw_dir, plan)
    require(sha256(tokenizer_model) == plan["tokenizer_sha256"], "Tokenizer SHA256 differs from the plan")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tasks = []
    for index, sources in enumerate([*groups, validation]):
        split, shard = ("train", index) if index < TRAIN_SHARDS else ("valid", 0)
        relative = f"{split}/fineweb_edu_100bt.{split}.{shard:03d}_text_document"
        tasks.append({"sources": [str(path) for path in sources], "tokenizer_model": str(tokenizer_model),
                      "output_prefix": str(output_dir / relative), "relative_prefix": relative})
    if workers == 1:
        records = [process_shard(task) for task in tasks]
    else:
        records = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_shard, task): index for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                records[futures[future]] = future.result()
    require(all(record and record["status"] == "complete" for record in records), "Some shards did not complete")
    require(sha256(plan_path) == plan_sha256, "Data plan changed during preprocessing")
    manifest = {"schema_version": 1, "status": "complete", "dataset_repo": plan["dataset_repo"],
        "dataset_revision": plan["dataset_revision"], "plan_sha256": plan_sha256,
        "tokenizer_sha256": plan["tokenizer_sha256"], "vocab_size": VOCAB_SIZE, "eod_id": EOD_ID,
        "encoding": "ordinary text; no BOS; 400000-character chunks; EOD after every document including empty text",
        "ordering": "fixed output groups and source order from plan; original rows within each Parquet",
        "training_sources": plan["training_sources"], "validation_source": plan["validation_source"],
        "shards": records}
    atomic_json(output_dir / "manifest.json", manifest)
    data = {"train": [{"weight": 1.0, "prefix": task["output_prefix"]} for task in tasks[:-1]],
            "validation": [{"weight": 1.0, "prefix": tasks[-1]["output_prefix"]}],
            "tokenizer_model": str(tokenizer_model)}
    atomic_json(output_dir / "data.json", data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan", type=Path, default=REPOSITORY / "configs/fineweb_edu_100bt.json")
    args = parser.parse_args()
    prepare(args.raw_dir, args.tokenizer_model, args.output_dir, args.plan, args.workers)
    print("Training data ready: " + str(args.output_dir.resolve() / "data.json"), flush=True)


if __name__ == "__main__":
    main()
