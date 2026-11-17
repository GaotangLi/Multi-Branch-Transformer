# Checkpoint conversion

The published checkpoints use **native Megatron tensor names in model-only `safetensors` bundles**. They contain model weights, not optimizer state or a distributed training-resume checkpoint.

This optional utility converts those release bundles into an **experimental Transformers reference**. It supports the released Standard models and Balanced Joint-KV MBTs with 2, 3, or 4 branches. It does not convert arbitrary Megatron DCP directories, download weights, or upload models.

## Convert a local bundle

Use a separate environment if your training environment has different dependency versions:

```bash
python -m pip install -r utils/checkpoint_conversion/requirements.txt
python utils/checkpoint_conversion/convert_to_hf.py \
  --input /path/to/native-release \
  --output /path/to/new-hf-export
```

The input must contain `config.json`, `manifest.json`, and `model.safetensors`. The output directory must be new and separate. Conversion runs on CPU; allow roughly three times the input weight size in host memory and enough disk space for another complete model. On a cluster, run it on a compute allocation and use large storage.

The converter verifies the input checksums, architecture, every tensor's key/shape/dtype, complete source coverage, and exact serialization roundtrip. It preserves native Q/gate/K/V row packing and FP32 router biases. The tied output embedding is the only permitted duplicate. `conversion.json` records tensor mappings and checksums, without local account paths or private training metadata.

The resulting folder contains the weights, HF configuration, generation settings, and three standalone model files. **A successful conversion verifies tensors; it does not certify equivalent runtime outputs.**

## Tokenizer

No tokenizer is downloaded or synthesized. If you already have the exact training-compatible HF fast tokenizer, include it explicitly:

```bash
python utils/checkpoint_conversion/convert_to_hf.py \
  --input /path/to/native-release \
  --output /path/to/new-hf-export \
  --tokenizer-dir /path/to/compatible-tokenizer
```

Retain its applicable license and notice files. The utility checks vocabulary size and special-token roles; the caller must supply the correct BPE vocabulary and merges. These checkpoints use the training tokenizer's special-token mapping and no automatic BOS/EOS insertion. A default Llama chat tokenizer is **not** an interchangeable substitute.

## Reference inference

With a compatible tokenizer included in the export:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

directory = "/path/to/new-hf-export"
tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    directory, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32,
).to("cuda").eval()
inputs = tokenizer("The future of language models", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The implementation uses PyTorch and Transformers, without Megatron or Transformer Engine. It supports `DynamicCache`, causal prefill, incremental decoding, and padded batches. Static, quantized, and offloaded caches and attention-map output are not implemented. This reference is not the optimized serving engine used for the paper's speed measurements.

## Validation scope

The tested software versions are PyTorch 2.12 and Transformers 5.15. Local FP32 cache/generation checks passed for all 16 selected models. Native-versus-portable IEEE FP32 graph checks passed for the first two tested models; that comparison is incomplete for the full release.

BF16 execution is **not bitwise equivalent** to native execution. Small operator differences can accumulate and alter logits or greedy predictions, including between cached and uncached execution. Use FP32 for reference checks; the native releases remain authoritative for the reported native results. The converter leaves their weights untouched.

Run the small synthetic mapping tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p 'test_checkpoint_conversion.py' -v
```

These tests use tiny in-memory tensors and no checkpoints, datasets, network access, or GPU. They cover layer/expert ordering, Joint-KV keys, bridges/fusion, FP32 router biases, tied embeddings, and fail-closed handling of missing, extra, or malformed tensors.
