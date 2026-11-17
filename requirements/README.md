# Environment profiles

Use Python 3.12 on Linux and install a CUDA-enabled PyTorch build appropriate
for the GPU and driver first. The native training path requires NVIDIA
Transformer Engine and a compatible FlashAttention backend; a CPU-only
PyTorch wheel is sufficient only for non-GPU development tasks.

The observed research environment is recorded in
[`reference-environment.json`](reference-environment.json). It is a provenance
snapshot, **not a portable lockfile**: CUDA libraries and compiled extensions
must agree with the installed PyTorch ABI and hardware. New environments
should pass the GPU smoke before starting a full run.

See [installation](../docs/installation.md) for setup and source-build commands.
The repository does not install the PyPI `megatron-core` package: its native
backend is included because the latent MoE and router changes are required.
