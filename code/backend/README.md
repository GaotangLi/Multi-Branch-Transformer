# Native training backend

This directory contains the selected Megatron-LM training source used by the
Multi-branch Transformer project, with its latent MoE and router changes.
Use the public entry point in [`code/mbt`](../mbt) and the
[training guide](../../docs/training.md), rather than treating this as a full
upstream Megatron distribution.

The complete original license is in [LICENSE](LICENSE). Input hashes and the
research revision are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json);
portable build changes are in [PORTABILITY.md](PORTABILITY.md).

No compiled kernels, model weights, or data are bundled. See
[third-party notices](../../THIRD_PARTY_NOTICES.md) for attribution.
