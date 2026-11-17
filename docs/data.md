# Data preparation

For evaluation of a released checkpoint, use the small
[MBT-Validation dataset](https://huggingface.co/datasets/gaotang/MBT-Validation)
and the [validation guide](../utils/VALIDATION.md). The steps below prepare the
full corpus for **training from scratch**.

## 1. Environment and storage

Use the [project environment](installation.md) and run from the repository root
on a compute node. Preparation uses CPU workers, not GPUs. Set the worker count
to fit your allocation. The original raw and indexed data together occupy about
678 GB; allow roughly 1 TB of free storage for preparation.

```bash
export MBT_DATA_ROOT=/scratch/your-user/mbt-data
export TMPDIR="$MBT_DATA_ROOT/tmp"
export HF_HOME="$MBT_DATA_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$MBT_DATA_ROOT/cache/pip"
export TIKTOKEN_CACHE_DIR="$MBT_DATA_ROOT/cache/tiktoken"
export MBT_DATASET_HELPER_CACHE="$MBT_DATA_ROOT/cache/dataset-helper"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"
python -m pip install pyarrow huggingface_hub
```

Replace the example root with your writable large-storage directory. Keep
downloaded data and generated files outside the Git checkout.

## 2. Download FineWeb-Edu and the tokenizer

We use [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
`sample-100BT`, pinned to revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The public
[preparation plan](../configs/fineweb_edu_100bt.json) records every source file
and its ordered output group. `000_00000.parquet` is held out for validation;
the other 139 files form 64 training shards. Training blends them equally in
the recorded order.

The tokenizer is Meta's original
[`original/tokenizer.model`](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main/original),
not a SentencePiece model or a chat-template tokenizer. Obtain access through
the Meta repository and authenticate with `hf auth login` if required. The
following downloads only the tokenizer file, not Llama model weights:

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

root = Path(os.environ["MBT_DATA_ROOT"])
plan = json.loads(Path("configs/fineweb_edu_100bt.json").read_text())

tokenizer_revision = HfApi().model_info(plan["tokenizer_repo"]).sha
tokenizer = Path(hf_hub_download(
    plan["tokenizer_repo"], plan["tokenizer_file"],
    revision=tokenizer_revision, local_dir=root / "tokenizer",
))
assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == plan["tokenizer_sha256"]
(root / "tokenizer/revision.txt").write_text(tokenizer_revision + "\n")

names = [plan["validation_source"]]
names += [name for group in plan["training_sources"] for name in group]
snapshot_download(
    plan["dataset_repo"], repo_type="dataset", revision=plan["dataset_revision"],
    allow_patterns=[plan["source_directory"] + "/" + name for name in names],
    local_dir=root / "raw", max_workers=4,
)
PY
```

FineWeb-Edu retains its ODC-By attribution; the tokenizer retains Meta's license
and access terms. Neither is redistributed in this code repository.

## 3. Build indexed training data

```bash
python utils/prepare_training_data.py \
  --raw-dir "$MBT_DATA_ROOT/raw/sample/100BT" \
  --tokenizer-model "$MBT_DATA_ROOT/tokenizer/original/tokenizer.model" \
  --output-dir "$MBT_DATA_ROOT/indexed" \
  --workers 8
```

The helper streams Parquet directly into Megatron `.bin` / `.idx` files;
no large JSONL copy is needed. It checks the tokenizer checksum and preserves
the plan's file and document order. Text is encoded without a BOS token or chat
template, then each document receives EOD `128001`, including empty documents.
Documents longer than 400,000 characters are encoded in consecutive chunks,
matching the original preparation. Vocabulary size is 128,256.

Use a **new output directory**. Existing outputs are never overwritten. A
successful run produces:

```text
indexed/
  train/fineweb_edu_100bt.train.000_text_document.bin  # through .063
  train/fineweb_edu_100bt.train.000_text_document.idx
  valid/fineweb_edu_100bt.valid.000_text_document.bin
  valid/fineweb_edu_100bt.valid.000_text_document.idx
  data.json
  manifest.json
```

`data.json` is written only after every shard is complete. It contains absolute
paths for this machine: 64 ordered training prefixes with weight `1.0`, one
validation prefix, and the tokenizer path. Prefixes omit `.bin` / `.idx`.
Keep the files and blend order unchanged when resuming a training run.

## 4. Inspect and train

```bash
PYTHONPATH=code python -m mbt.train \
  --config configs/w1/balanced-2-joint.json \
  --data-config "$MBT_DATA_ROOT/indexed/data.json" \
  --output-dir "$MBT_DATA_ROOT/runs/w1-balanced-2-joint" \
  --world-size 8 --expert-parallel-size 8 --dry-run
```

On an allocated eight-GPU node:

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  --config configs/w1/balanced-2-joint.json \
  --data-config "$MBT_DATA_ROOT/indexed/data.json" \
  --output-dir "$MBT_DATA_ROOT/runs/w1-balanced-2-joint" \
  --temp-root "$TMPDIR" --expert-parallel-size 8
```

Training constructs 4,096-token causal samples from the indexed documents and
evaluates a fixed validation prefix. For the exact published checkpoint
comparison, use [MBT-Validation](../utils/VALIDATION.md), which already preserves
the original evaluation samples. See [training.md](training.md) for model
recipes, logging, checkpointing and resume.

<details>
<summary>Use your own JSONL corpus instead</summary>

Provide one JSON object per document with a `text` field, and separate training
and validation files. In the project environment:

```bash
export PYTHONPATH="$PWD/code/backend${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MBT_DATA_ROOT/custom-indexed"
python code/backend/tools/preprocess_data.py \
  --input /scratch/your-data/train.jsonl --json-keys text \
  --tokenizer-type TikTokenizer \
  --tokenizer-model "$MBT_DATA_ROOT/tokenizer/original/tokenizer.model" \
  --vocab-size 128256 --tiktoken-num-special-tokens 256 --append-eod \
  --output-prefix "$MBT_DATA_ROOT/custom-indexed/train" --workers 8
```

Repeat for a separate validation file. Copy
[`configs/data.example.json`](../configs/data.example.json), then point it at
the resulting `train_text_document` and validation prefixes plus the tokenizer.
This generic JSONL route is for custom data; the helper above preserves the
paper's FineWeb-Edu preparation rules.

</details>
