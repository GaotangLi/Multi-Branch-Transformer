# Training

The public entry point constructs an explicit `MBTModel` and trains it with Megatron's native attention, Quantile Balancing router, LatentMoE, and optimizer. The model owns its prelude, branches, fusion, and coda; an explicit training lifecycle handles validation and checkpoint retention. Model construction does not patch backend forwards or select a process-global experiment case.

## Recipes

`configs/w1` through `configs/w4` contain 44 explicit recipes:

- `standard.json`: sequential Standard.
- `wide-shallow-{2,3,4}.json`: independently parameterized, native-width branches.
- `narrow-deep-2.json`: narrower branches retaining the Standard path depth.
- `balanced-{2,3,4}-{independent,joint}.json`: intermediate branch width/depth; Joint-KV shares one cache per aligned branch depth.

All dimensions are explicit: `hidden_size` is residual width **d**, `latent_size` is the shared MoE latent width **m**, and `ffn_hidden_size` is each expert's SwiGLU inner width **f**. `branch_*` applies these widths inside branches. There are 64 experts, top-8 routing, GQA with two query heads per KV head, and head width 64. The shared initial/final stacks use the outer widths. Fusion is normalized concatenation followed by one projection, without an additional residual or MoE.

| Scale | Standard blocks | Outer d / m / f | Global batch | Updates | Training tokens |
|---|---:|---:|---:|---:|---:|
| W1 | 12 | 1024 / 512 / 512 | 128 | 53,110 | 27,844,935,680 |
| W2 | 14 | 1152 / 576 / 576 | 128 | 78,370 | 41,088,450,560 |
| W3 | 16 | 1280 / 640 / 640 | 256 | 55,258 | 57,942,212,608 |
| W4 | 18 | 1408 / 704 / 704 | 256 | 75,189 | 78,841,380,864 |

Every recipe uses 4,096-token sequences and its scale's same data horizon. Standard recipes use LR5e-4; branch examples use LR7e-4 for W1/W2 and LR5e-4 for W3/W4. These are runnable configurations, not a claim that each file names the best checkpoint from the LR sweep. The paper's shape controls and branch-count study sometimes use different allocations; the released Wide–Shallow recipes follow the approximately 50% branch-count study (including asymmetric W2/W4 branches).

## Inspect, then launch

Follow the [data preparation guide](data.md) for the pinned FineWeb-Edu download, tokenizer, and indexed shards. Pass the generated `data.json` as `--data-config`, replacing the example path below. A dry run imports no GPU packages, writes no experiment files, and prints the full model/training/data contract and native arguments:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code python -m mbt.train \
  --config configs/w1/balanced-2-joint.json \
  --data-config /scratch/data.json \
  --output-dir /scratch/runs/w1-balanced-2-joint \
  --world-size 8 --expert-parallel-size 8 --dry-run
```

On an allocated node, after installing the CUDA environment:

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  --config configs/w1/balanced-2-joint.json \
  --data-config /scratch/data.json \
  --output-dir /scratch/runs/w1-balanced-2-joint \
  --temp-root /scratch/tmp --expert-parallel-size 8
```

The model supports TP=PP=CP=expert-TP=1, EP=1/2/4/8, with data parallelism across the remaining ranks. The release examples target one node; the optional multi-node torchrun wiring is not a qualified cross-node training recipe. There is no CUDA-stream branch parallelism in this training implementation. Eight GPUs with MBS16 give GA1 for W1/W2 and GA2 for W3/W4. Memory capacity depends on the GPU and layout; `--micro-batch-size 8` doubles accumulation without changing the global batch or token horizon. The launcher never changes batch size or silently retries after an error.

`scripts/slurm/train.sbatch` is a single-node example. Set your own partition/account/time/GPU/CPU/memory and scratch stdout/stderr paths in `sbatch`. It contains no reservation, node blacklist, local paths, credential, or automatic cancellation/resubmission logic. The script does not submit itself.

## Validation and checkpointing

The paper recipe evaluates a fixed validation prefix of 12,160 sequences (49,807,360 positions) at approximately every 10% of training and at the final update. The same prefix is reused after a resume. Validation preserves train data counters, Python/NumPy/PyTorch/model-parallel RNG, module modes, and Quantile Balancing state. It records `validation.jsonl` alongside native TensorBoard metrics.

Native `torch_dist` saves include model, Adam optimizer, scheduler, RNG, and consumed-sample progress. The model format is `mbt-hierarchical-v2`, with explicit prelude/branch/coda keys recorded in a schema-2 run manifest. Saves are synchronous. By default, keep the latest **one complete checkpoint**; change `keep_last_checkpoints` in the JSON or pass `--keep-last-checkpoints N`. A new save must complete before an old one is removed. Only unchanged checkpoints with this run's completion record are eligible for retention; foreign or partial artifacts are preserved. The launcher checks a conservative retention-plus-next-save free-space estimate, but cannot reserve storage or infer your cluster's quota.

Use exactly the same command with `--resume` to continue the output directory's latest complete checkpoint. Model, optimizer schedule, data blend, world size/EP, and logging identity must match its stored manifest. A completed training run is rejected instead of restarting. Changing scientific settings requires a new output directory; this launcher does not provide implicit fine-tuning or weights-only resume.

Historical adapter checkpoints are not accepted as full-state resumes. `mbt.model.convert_legacy_state_dict` explicitly renames a **raw model state dictionary** for a matching configuration, leaving tensor values unchanged. This is a weights-only utility; it does not migrate distributed optimizer state, data progress, or the run manifest. See the [reference parity test](testing.md) for strict same-weight loading and comparison.

`--stop-after-step N` is an explicit early stop for mechanics/debugging, preserving the original schedule. `--smoke-steps N` instead creates a clearly named mock-data test with its own short schedule and sequence length; never use that option for paper training or reuse its output directory for a formal run.

## Storage and optional W&B

Choose an absolute output path on scratch/large storage. Checkpoints, dataset index cache, TensorBoard, W&B, compiler caches and temporary files stay there. A short scratch `--temp-root` avoids Unix socket path limits; compiler caches are attempt/rank isolated. Dataset helper builds share an attempt-local cache across ranks. An explicitly set `HF_HOME` is preserved. No weights, data or tokenizer are downloaded automatically.

W&B is opt-in. Supply all of `--wandb-entity`, `--wandb-project`, `--wandb-run-id`, and optionally `--run-name`. Credentials come from your normal W&B login/environment. Required logging must initialize online before training, and failure stops the job. Names/IDs accept 1–120 ASCII letters, digits, dots, underscores and hyphens. The launcher clears inherited `WANDB_SERVICE`/`_WANDB_SERVICE` sockets at startup and uses the same run ID with strict resume semantics.

## Tests

CPU contract, safe-retention, and numerical decoder tests (a CPU PyTorch build is sufficient):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code python -m unittest discover -s tests -p 'test_*.py' -v
```

One allocated GPU, using the training environment:

```bash
MBT_TEST_ROOT=/scratch/tests/mbt-native-unique \
MBT_TEMP_ROOT=/scratch/tmp bash scripts/check_native.sh
```

This constructs a small explicit Balanced Joint-KV model with the native E64/top8/head64 operators, checks future-token causality, trains two updates, saves a complete checkpoint, and resumes to update four. It does not certify paper-size memory capacity, EP8 throughput, or cached serving. The script fails closed on nonfinite losses, a missing complete checkpoint, or any restore error. Use a fresh test root for each attempt.
