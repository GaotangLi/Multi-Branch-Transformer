# Source provenance

The native attention, latent MoE, expert parallelism, Quantile Balancing,
optimizer and distributed checkpoint components originate from research
revision `1b241fc3f9da40d5b23ace37d34ae69cd1465fcd`. Replacing them with an
unmodified `megatron-core` installation changes the recipe.

The public MBT implementation composes these components directly:

| Component | Implementation |
|:---|:---|
| Backbone and branches | Explicit `MBTBlock`, `MBTDecoder`, and `MBTModel` classes |
| Fusion and width changes | Tokenwise RMSNorm and bias-free linear projections |
| Joint-KV | Explicit shared K/V producers and branch-local attention consumers |
| Pretraining | MBT model/data providers passed to the native training loop |
| Validation and retention | Per-run lifecycle callbacks, with strict full-state resume |

The previous MBT hook adapters and their global model registry are removed.
The arithmetic and parameter initialization scales are preserved; construction
order can change random-number consumption. Numerical parity tests therefore
load identical weights before comparing outputs and gradients. Hierarchical
checkpoint keys identify the new model format. A weights-only key converter
does not migrate historical optimizer checkpoints.

[`SOURCE_MANIFEST.json`](../code/backend/SOURCE_MANIFEST.json) records the
original backend extraction inputs, including subsequently removed entry
points. [`PORTABILITY.md`](../code/backend/PORTABILITY.md) lists release changes;
the current Git revision identifies the released source.

The repository excludes institutional paths, credentials, historical batch
jobs, training data, model weights and runtime outputs. Native dependency
modules outside the public recipe remain where required by imports. Supported
training uses TP=PP=CP=1 and EP=1/2/4/8.

Paper results come from full research runs. New public-code CPU tests, GPU
mechanics tests and full-layout training checks are reported separately in
the [validation record](validation.md).
