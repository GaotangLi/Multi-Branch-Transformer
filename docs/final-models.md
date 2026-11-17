# Final Balanced Joint-KV models

These are the final 2-, 3-, and 4-branch models highlighted on the homepage. W1–W4 denote the four model scales. Each comparison uses the same per-scale training-token budget and validation protocol; lower loss is better.

## Configurations and validation loss

Allocations are `prelude / (branch depths) / coda`. Dimensions are branch residual width **d**, MoE latent width **m**, and expert hidden width **f**. Shared outer blocks retain the Standard widths. KV savings are theoretical full-model reductions relative to the same-scale Standard at matched context, batch size, and cache precision.

| Scale | Branches | Allocation | Branch d / m / f | Standard loss | MBT loss | KV saved | Configuration |
|:---|---:|:---|:---|---:|---:|---:|:---|
| W1 | 2 | 3 / (5 + 5) / 3 | 768 / 368 / 384 | 2.653 | **2.620** | 18.75% | [JSON](../configs/w1/balanced-2-joint.json) |
| W1 | 3 | 3 / (4 + 4 + 4) / 3 | 640 / 368 / 368 | 2.653 | **2.622** | 29.17% | [JSON](../configs/w1/balanced-3-joint.json) |
| W1 | 4 | 3 / (3 + 3 + 3 + 3) / 3 | 640 / 368 / 368 | 2.653 | **2.633** | 34.38% | [JSON](../configs/w1/balanced-4-joint.json) |
| W2 | 2 | 3 / (6 + 6) / 3 | 896 / 464 / 464 | 2.554 | **2.528** | 23.81% | [JSON](../configs/w2/balanced-2-joint.json) |
| W2 | 3 | 3 / (5 + 5 + 5) / 4 | 768 / 352 / 352 | 2.554 | **2.537** | 26.19% | [JSON](../configs/w2/balanced-3-joint.json) |
| W2 | 4 | 3 / (4 + 4 + 4 + 4) / 4 | 640 / 384 / 384 | 2.554 | **2.534** | 34.13% | [JSON](../configs/w2/balanced-4-joint.json) |
| W3 | 2 | 4 / (6 + 6) / 4 | 1024 / 512 / 512 | 2.471 | **2.434** | 20.00% | [JSON](../configs/w3/balanced-2-joint.json) |
| W3 | 3 | 4 / (5 + 5 + 5) / 4 | 896 / 432 / 448 | 2.471 | **2.451** | 28.13% | [JSON](../configs/w3/balanced-3-joint.json) |
| W3 | 4 | 4 / (4 + 4 + 4 + 4) / 4 | 896 / 400 / 400 | 2.471 | **2.454** | 32.50% | [JSON](../configs/w3/balanced-4-joint.json) |
| W4 | 2 | 4 / (7 + 7) / 4 | 1152 / 592 / 592 | 2.392 | **2.361** | 23.74% | [JSON](../configs/w4/balanced-2-joint.json) |
| W4 | 3 | 4 / (6 + 6 + 6) / 5 | 896 / 496 / 496 | 2.392 | **2.366** | 28.79% | [JSON](../configs/w4/balanced-3-joint.json) |
| W4 | 4 | 4 / (6 + 6 + 6 + 6) / 5 | 768 / 432 / 432 | 2.392 | **2.364** | 31.82% | [JSON](../configs/w4/balanced-4-joint.json) |

The JSON files specify model dimensions and runnable training recipes. Data paths and tokenizer files must be provided separately; checkpoint downloads are on hold. For initialization, optimizer, validation and resume details, see [training.md](training.md).

## Fitted scaling comparisons

The figure fits `loss = A × C^(-alpha)`, with **C** measured in training-body PFLOP-days. The fitted compute ratio compares Standard and MBT at validation loss **2.500**, within their common measured ranges.

| Model | A | alpha | Standard / MBT body compute at loss 2.500 |
|:---|---:|---:|---:|
| Standard | 2.5204 | 0.05134 | 1.00× |
| Final 2-branch MBT | 2.4908 | 0.05221 | 1.26× |
| Final 3-branch MBT | 2.4988 | 0.05149 | 1.18× |
| Final 4-branch MBT | 2.5013 | 0.05302 | 1.16× |

The equal-loss ratio is fitted, not a measurement of training wall time. Full-precision [loss values](../assets/final-model-losses.csv) and [fit coefficients](../assets/final-model-scaling.json) accompany the figure.

## Inference comparisons

The homepage frontier fixes **one H200, 16K input tokens, 1K output tokens, and 64 concurrent requests**, with all models served by the same engine. Its quality axis uses validation perplexity measured at the 4,096-token training context, not a long-context quality evaluation.

The peak **51.4% decoding throughput improvement** is a separate W2 4-branch setting: **one H200, 64K input tokens, 1K output tokens, and 8 concurrent requests**. It compares decode-only output throughput with the same-scale Standard. The up-to **34% KV saving** rounds 34.38% and concerns cache storage rather than total GPU memory.

The public package here provides the training model and tests. The measured inference engine is separate from these training launchers.
