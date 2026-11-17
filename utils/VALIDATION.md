# Validate a native checkpoint

`utils/evaluate_native.py` measures token-weighted next-token cross entropy and
perplexity on one CUDA GPU. It loads a published native bundle through
`utils/native_checkpoint.py`; no optimizer state, tokenizer or KV cache is
needed. Install the [native dependencies](../docs/installation.md) and
[download a checkpoint](NATIVE_CHECKPOINTS.md) first.

## Download the fixed validation set

[MBT-Validation on Hugging Face](https://huggingface.co/datasets/gaotang/MBT-Validation)
contains **12,160 fixed sequences** as Parquet: one `input_ids` column with
**4,097 Llama 3 token IDs per row**. These are the paper's held-out FineWeb-Edu
samples in their original order. No tokenizer or full training corpus is needed.

Prepare them on a compute node, using your scratch/large-storage paths:

```bash
python -m pip install numpy pyarrow huggingface_hub
python utils/prepare_validation.py \
  --cache-dir /scratch/cache/huggingface/hub \
  --output /scratch/validation
```

The script downloads a pinned dataset revision, checks the token checksum,
and creates `tokens.npy` plus `manifest.json` in a new directory. It copies
IDs without retokenizing, repacking or shuffling. Every target is scored, so
no mask file is needed. An existing downloaded Parquet file can instead be
passed with `--parquet /scratch/validation.parquet`.

The evaluator uses each row's first 4,096 IDs as inputs and last 4,096 IDs
as labels, shifting exactly once. It applies ordinary causal attention;
positions and attention do not reset at document boundaries.

## Run

From the repository root, with one allocated GPU and large scratch storage:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code \
torchrun --standalone --nproc_per_node=1 utils/evaluate_native.py \
  --bundle /scratch/MBT-2-branch-W1 \
  --validation-data /scratch/validation \
  --batch-size 8 \
  --output /scratch/eval/MBT-2-branch-W1.json \
  --temp-root /scratch/tmp
```

Use a fresh output path for each model or attempt. Add `--max-samples 8` for an
explicit prefix smoke test; its result is not the full validation loss. Add
`--per-sequence-losses /scratch/eval/MBT-2-branch-W1-losses.npy` when individual
sequence losses are needed. Data, weights, logs and caches belong on storage
appropriate for model evaluation.

Original evaluation microbatch size was **2 for Standard-W1/W2/W3**, and **1 for
all MBTs and Standard-W4**. Use those batch sizes when comparing with the
original measurements. The original distributed evaluation used eight GPUs;
this one-GPU evaluation can differ numerically because of BF16 kernels and
reduction order.

The JSON result contains validation loss, perplexity, sample/token counts,
and input checksums.

<details>
<summary>Prepare a manifest for existing token arrays</summary>

For other pretokenized samples, use `tokens.npy` with shape `[N, S + 1]`,
dtype `int32` or `int64`, and context length `S` up to 4,096. An optional
binary `loss_mask.npy` of shape `[N, S]` selects targets; without it every
target is scored. Create the manifest on a compute node:

```python
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path("/scratch/validation")

def entry(filename):
    path = root / filename
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return {"file": filename, "sha256": digest.hexdigest(),
            "shape": list(array.shape), "dtype": str(array.dtype)}

tokens = entry("tokens.npy")
assert len(tokens["shape"]) == 2 and tokens["dtype"] in ("int32", "int64")
n, width = tokens["shape"]
assert n > 0 and 1 <= width - 1 <= 4096
manifest = {
    "schema_version": 1,
    "context_length": width - 1,
    "num_sequences": n,
    "tokens": tokens,
    "provenance": {"description": "Existing frozen validation samples"},
}
if (root / "loss_mask.npy").exists():
    mask = entry("loss_mask.npy")
    assert mask["shape"] == [n, width - 1]
    manifest["loss_mask"] = mask
with (root / "manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
```

Keep an existing verified manifest instead of regenerating it. For paper
replication, preserve the original dataset/index construction, tokenizer,
document boundaries, sample order and loss masks; hashing a different packing
does not make it the original validation set.

</details>

Run the small accounting tests in an environment with NumPy, PyArrow and PyTorch:

```bash
PYTHONDONTWRITEBYTECODE=1 python utils/test_evaluate_native.py -v
PYTHONDONTWRITEBYTECODE=1 python utils/test_prepare_validation.py -v
```
