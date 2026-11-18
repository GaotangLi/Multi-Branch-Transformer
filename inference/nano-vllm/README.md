# MBT Serving on Nano-vLLM

A single-GPU inference engine for **multi-branch Transformers (MBT)** with
**shared KV caches**, built on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

An MBT model runs a few *prelude* layers, splits into 2–4 parallel *branches*
(each a stack of layers), concatenates the branches, and finishes with a few
*coda* layers. Branch layers can share their keys and values:

- **Prelude-KV**: one KV bank, computed once from the prelude output, is read by
  every branch layer.
- **Joint-KV**: one KV bank per branch depth, computed from all branches'
  normalized inputs at that depth.

In this directory we release:

- **Inference engine**
  - Paged KV cache, FlashAttention-3/4 attention, Triton MoE and fused
    elementwise kernels, CUDA-graph decode and FlashInfer sampling.
  - *Batched* branch execution: all branches at the same depth run as one
    stacked computation, and their query heads are folded into one GQA group per
    shared KV head, so each KV bank is read once per depth.
  - Batched execution is **bitwise identical** to running the branches one by
    one (*serial*).
- **Benchmark**
  - A fixed-concurrency throughput benchmark for one exported model, which first
    checks serial vs. batched outputs bitwise, then measures throughput per mode.

## Table of Contents

- [Code Organization](#code-organization)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Using the Published Weights](#using-the-published-weights)
  - [Running Without a Trained Checkpoint](#running-without-a-trained-checkpoint)
  - [Model Exports](#model-exports)
  - [Prompt Manifests](#prompt-manifests)
- [Benchmark](#benchmark)
  - [One Model](#one-model)
  - [Decode Speedup over a Standard Transformer](#decode-speedup-over-a-standard-transformer)
- [Tests](#tests)
- [Fixed Choices](#fixed-choices)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Code Organization

- [`nanovllm/experimental/wide_serving/`](nanovllm/experimental/wide_serving) :
  The base engine and all kernels. Runs version-1 exports (Standard Transformer
  and two-branch Concat models).
  - `model.py` model layers · `runner.py` chunked prefill, CUDA-graph decode
    and sampling · `attention.py` paged KV cache and FA3/FA4 attention ·
    `moe.py`, `ops_router.py` MoE and routing · `ops.py`, `ops_qk.py` Triton
    kernels · `export.py` checkpoint export · `config.py` model configuration.
- [`nanovllm/experimental/mbt_serving/`](nanovllm/experimental/mbt_serving) :
  Version-2 MBT models (2–4 branches, unequal depths) and shared KV. Reuses the
  `wide_serving` attention, MoE and kernels.
  - `model.py` input bridge, branches, KV banks (`KVBank`,
    `SharedKVAttention`) and batched execution · `runner.py` MBT runner ·
    `config.py` branch layout and `kv_mode` · `export.py` checkpoint export.
- [`benchmarks/mbt/`](benchmarks/mbt) : The throughput benchmark behind
  [`run_bench.py`](run_bench.py).
  - `runtime.py` arguments, parity check, throughput measurement and results ·
    `closed_loop/` fixed-concurrency scheduler and token accounting ·
    `prompt_manifest.py` prompt loader · `profiling.py`, `trace_analysis.py`
    statistics and optional profiling.
- [`tools/`](tools) : `convert_native_bundle.py` turns a published native
  bundle into a servable export · `make_demo_export.py` writes a bundle with
  synthetic random weights · `build_prompt_manifest.py` writes a prompt manifest
  from a corpus or from synthetic token ids.
- [`tests/`](tests) and `benchmarks/mbt/test_*.py` : Unit tests.
- `nanovllm/engine`, `nanovllm/layers`, `nanovllm/models`, `nanovllm/utils` :
  Upstream Nano-vLLM, unchanged.

The benchmark picks the engine from `format_version` in the export's
`config.json`: version 1 runs on `wide_serving`, version 2 on `mbt_serving`.

## Getting Started

### Installation

Tested on NVIDIA H200 with Python 3.10, PyTorch 2.13 (CUDA 12.9), Triton 3.7,
`sglang-kernel` 0.4.6 (FlashAttention-3), `flashinfer-python` 0.6.17,
`flash-attn-4` 4.0.0b19 and `safetensors` 0.8.

```bash
conda create -n mbt-serving python=3.10 pip -y
conda activate mbt-serving
python -m pip install torch triton numpy safetensors \
  sglang-kernel flashinfer-python flash-attn-4
```

> **Note:** this environment setup has not been verified on a fresh install.
> `sglang-kernel`, `flashinfer-python` and `flash-attn-4` must be builds that
> match your PyTorch and CUDA versions. Python 3.10 is the only version the
> tests have been run on.

No installation of this directory is needed; `run_bench.py` imports it from
the checkout. `pip install .` is available for the import path alone and does
not pull in the GPU kernels; `pip install '.[wide]'` names them.

### Using the Published Weights

The trained models are released as Megatron-native model-only bundles
(`config.json`, `manifest.json`, `model.safetensors`) for full-sequence loading.
`tools/convert_native_bundle.py` rewrites one into an `mbt-serving-v2` export
that this engine serves with a KV cache. Tensor names are mapped, experts
stacked or layers sliced, and nothing is recomputed:

```bash
pip install huggingface_hub
hf download gaotang/MBT-4-branch-W1 --local-dir MBT-4-branch-W1
hf download gaotang/Standard-W1 --local-dir Standard-W1

python tools/convert_native_bundle.py --bundle MBT-4-branch-W1 --output mbt-4-branch-w1
python tools/convert_native_bundle.py --bundle Standard-W1 --output standard-w1
```

Both published layouts are recognized from the bundle's own manifest:
`mbt-native-shared-kv-v1` for the Prelude-KV and Joint-KV multi-branch models,
`mbt-native-standard-global-v1` for the Standard baselines. The export records
the source bundle's name and the SHA-256 of the exact `config.json` and
`model.safetensors` it came from, so a measurement can be traced to a published
checkpoint. The two exports above are the pair the next section measures.

See [`utils/NATIVE_CHECKPOINTS.md`](../../utils/NATIVE_CHECKPOINTS.md) for the
bundles themselves and the loader that reads them for full-sequence work.
Prompts still have to be tokenized with the matching tokenizer, which the
bundles do not carry; the models were trained at context 4,096 with the Llama-3
vocabulary of 128,256 tokens. Any run whose prompt plus generated tokens exceeds
4,096 needs `--system-only-extrapolation`, which is explained under
[One Model](#one-model).

### Running Without a Trained Checkpoint

Without downloading anything, two tools build a bundle and a prompt manifest
out of nothing, so the engine, the parity check and the benchmark can all be
exercised on one GPU:

```bash
python tools/make_demo_export.py --output demo-joint-k4 --variant mbt --kv-mode joint-branch-kv
python tools/make_demo_export.py --output demo-standard --variant standard
python tools/build_prompt_manifest.py --output demo-L4096.jsonl \
  --length 4096 --count 256 --export demo-joint-k4

python run_bench.py --phase smoke --variant baseline --export demo-joint-k4 \
  --modes serial,batched --prompt-manifest demo-L4096.jsonl \
  --prompt-length 4096 --output-length 1024 --concurrency 8 \
  --prefill-token-budget 8192 --sampling greedy --system-only-extrapolation \
  --output-dir smoke-joint-k4
```

The weights are random and the manifest holds random token ids, so generated
text is meaningless. Everything else is real: the bundle goes through the same
strict loader, and `--modes serial,batched` still proves bitwise equality of
the two execution paths. Throughput measured this way is also real, because a
dense-attention forward pass does the same work whatever the token values are —
only claims about output quality need a trained model and real prompts. The
bundle's `manifest.json` records `trained: false` and the manifest records
`source: synthetic-random` so such a run cannot be mistaken for a measurement
of a trained model.

### Model Exports

The engine loads an export directory with `config.json`, `manifest.json` and
`model.safetensors`. Standard and two-branch models are exported with
`nanovllm/experimental/wide_serving/export.py`; MBT and shared-KV models with
`nanovllm/experimental/mbt_serving/export.py`. Both convert a native Megatron
distributed checkpoint and need the proof file written beside it; `--scratch-root`
is optional and confines the output to one directory tree. To convert a published
bundle instead, which needs no proof, use
[`tools/convert_native_bundle.py`](tools/convert_native_bundle.py).

### Prompt Manifests

Prompts are read from a JSONL manifest, one pre-tokenized prompt per line:

```json
{"id": "ruler-L65536-0", "source": "ruler", "length": 65536, "token_ids": [128000, ...], "tokenizer_sha256": "..."}
```

The first `C × prompt-rounds` records with the requested length are used, in
file order. Nothing is padded or truncated; any mismatch is an error. See
[`benchmarks/mbt/prompt_manifest.py`](benchmarks/mbt/prompt_manifest.py) for the
loader and [`tools/build_prompt_manifest.py`](tools/build_prompt_manifest.py) to
write one. `--expected-tokenizer-sha256` is optional; when given, every selected
prompt must carry that tokenizer digest.

## Benchmark

### One Model

`--output-dir` is the only path the benchmark writes to. On a cluster whose home
quota is too small for compiler caches, set `MBT_SCRATCH_ROOT`; the
benchmark then refuses to start unless the output directory and the four cache
directories below are all inside it:

```bash
export MBT_SCRATCH_ROOT=/path/to/scratch   # optional
export TMPDIR=$MBT_SCRATCH_ROOT/tmp
export TRITON_CACHE_DIR=$MBT_SCRATCH_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$MBT_SCRATCH_ROOT/inductor
export CUDA_CACHE_PATH=$MBT_SCRATCH_ROOT/cuda
mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $CUDA_CACHE_PATH

python run_bench.py --phase throughput --variant baseline \
  --export /path/to/mbt-export \
  --modes serial,batched \
  --prompt-manifest /path/to/prompts-L4096.jsonl \
  --prompt-length 4096 --output-length 1024 --concurrency 64 \
  --prefill-token-budget 8192 \
  --sampling sample --temperature 1.0 --top-p 0.95 --top-k 0 --sampling-seed 0 \
  --system-only-extrapolation \
  --output-dir $MBT_SCRATCH_ROOT/runs/mbt-L4096-C64
```

`--system-only-extrapolation` is needed here because 4,096 prompt tokens plus
1,024 generated tokens exceed the 4,096-token context these models were trained
at. The flag changes nothing about the computation; it records that the run is a
systems measurement and that output quality beyond the trained context is not
claimed. Every measurement we report was run with it.

The run first checks that serial and batched execution produce identical
tokens, logits and expert choices, then measures each mode for three repeats of
at least 60 seconds. `results.jsonl` ends with a `summary` record whose
`decode_by_mode` holds the decode throughput (tokens/s) of each mode.

- `--modes`: `serial` for a Standard model; `serial,batched` for a shared-KV
  MBT model.
- `--concurrency`: number of sequences decoded together (1, 2, 4, …, 128).
- `--prefill-token-budget`: tokens per prefill step, shared by all sequences.
- `--system-only-extrapolation`: required when prompt plus output exceeds the
  model's trained context (for example 64K prompts on a 4K-context model).

### Decode Speedup over a Standard Transformer

Run on one GPU, in this order, with identical arguments except `--export` and
`--modes`:

1. the Standard model, `--modes serial` (AR a);
2. each MBT model, `--modes serial,batched`;
3. the Standard model again, `--modes serial` (AR b).

The speedup of an MBT model is its `batched` decode throughput divided by the
mean of AR a and AR b, which cancels slow drift of the GPU during the runs.

## Fixed Choices

Two numerical choices are compiled into the benchmark path rather than exposed:

- The fused router kernel the benchmark selects (`router_backend='fused_post8'`)
  accepts exactly 64 experts, top-8 routing and `router_renorm_sum=2.5`. A
  bundle with any other routing shape must use another router backend.
- The Triton MoE tile sizes were picked by a sweep on an H200
  ([`moe.py`](nanovllm/experimental/wide_serving/moe.py)). They are correct on
  any GPU the kernels support and tuned for that one.

Otherwise the engine reads no tuning knobs: the only environment variables it
looks at are `WORLD_SIZE` (which must be 1), `CUDA_VISIBLE_DEVICES`, the
optional `MBT_SCRATCH_ROOT`, and `SLURM_JOB_ID` if present, which is
recorded for provenance.

## Tests

```bash
python -m pip install pytest
bash run_cpu_tests.sh            # benchmark tests and tests/test_wide_*.py, CPU only
python -m pytest tests           # engine tests
```

`run_cpu_tests.sh` takes an optional interpreter path (default `python3`) and
uses `$TMPDIR` when set, rejecting one on a home filesystem; otherwise it makes
and removes its own temporary directory.

Two tests in `tests/test_wide_model.py` check the router and QKV split against
the training implementation they were ported from, which this repository
vendors. Point `MBT_NATIVE_SOURCE` at it to run them:

```bash
MBT_NATIVE_SOURCE=../../code/backend python -m pytest -q tests/test_wide_model.py
```

GPU tests in `tests/test_mbt_model.py` run only with `MBT_CUDA_TESTS=1`, and the
tensor tests in `tests/test_mbt_export.py` only with `MBT_TORCH_TESTS=1`;
otherwise they are skipped. One test compares RMSNorm and RoPE against
Transformer Engine and skips unless it is installed (`pip install '.[parity]'`).

## License

MIT, for everything in this directory. Two copyright holders appear in
[`LICENSE`](LICENSE): `nanovllm/{engine,layers,models,utils}/`, `nanovllm/config.py`,
`nanovllm/llm.py` and `nanovllm/sampling_params.py` are Nano-vLLM's and are
unmodified; `nanovllm/experimental/`, `benchmarks/`, `tests/`, `tools/` and
`run_bench.py` are ours and carry an SPDX header. No Megatron-LM code is
included: the exporters read a Megatron checkpoint, and the two tests that
compare against the training implementation read it from a checkout you point
`MBT_NATIVE_SOURCE` at.

## Acknowledgements

This engine builds on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
(MIT License, see [LICENSE](LICENSE)). FlashAttention-3 kernels come from
[`sglang-kernel`](https://github.com/sgl-project/sglang) and sampling from
[FlashInfer](https://github.com/flashinfer-ai/flashinfer).
