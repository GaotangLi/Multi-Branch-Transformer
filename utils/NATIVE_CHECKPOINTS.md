# Native model checkpoints

The [Hugging Face collection](https://huggingface.co/collections/gaotang/multi-branch-transformer)
contains 12 Balanced Joint-KV MBTs (2/3/4 branches at W1–W4) and four Standard
references. Each bundle contains model-only `safetensors`, architecture metadata,
and checksums. The native tensor format is preserved; these files are not
Transformers `AutoModel` bundles or optimizer/resume checkpoints.

The public loader `utils/native_checkpoint.py` constructs the matching model
using this repository's Megatron backend. It supports both published native
formats, verifies checksums and every model tensor, and requires no private
training source, training data, or tokenizer download.

Install the native GPU dependencies in [the installation guide](../docs/installation.md),
then download a model to your large storage volume:

```python
from huggingface_hub import snapshot_download

snapshot_download("gaotang/MBT-2-branch-W1", local_dir="/scratch/MBT-2-branch-W1")
```

Inspect the model configuration without importing GPU libraries:

```bash
PYTHONPATH=code python utils/native_checkpoint.py \
  --bundle /scratch/MBT-2-branch-W1 --inspect
```

On one allocated CUDA GPU, run a full-sequence forward with explicit token IDs:

```bash
PYTHONPATH=code torchrun --standalone --nproc_per_node=1 \
  utils/native_checkpoint.py --bundle /scratch/MBT-2-branch-W1 \
  --token-ids 128000,791,3925,315,8198,369,264,2115 \
  --report /scratch/native-load-result.json
```

The same command supports `Standard-W1` through `Standard-W4`. The output reports
strict tensor-loading coverage, logits shape and the final position's top token
IDs. For text inputs, obtain the original Llama 3 tokenizer separately under its
applicable access and license terms.

In an already initialized single-GPU NCCL/Megatron process, `load_model(bundle)`
returns `(model, receipt)`. Use `model(input_ids, position_ids, None)` to compute
full-sequence logits of shape `[batch, tokens, vocabulary]`. The loader requires
TP=PP=EP=CP=1 and BF16 execution while retaining the router bias in FP32.
This path does not implement cached decoding or the specialized serving engine
used for the paper's throughput measurements.

For validation loss on a frozen set of pretokenized samples, see the
[validation guide](VALIDATION.md).

Small loader-contract tests require no GPU or Torch import:

```bash
PYTHONPATH=code python utils/test_native_checkpoint.py
```
