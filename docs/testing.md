# Testing the explicit model

The tests cover three different questions. A pass at one level does not substitute for the others.

| Check | What it verifies | Runtime |
|:---|:---|:---|
| CPU contracts and decoder | Configuration validity; branch decomposition; outputs and gradients; width changes; synchronized Joint-KV inputs; fusion without an extra residual | PyTorch CPU |
| Native smoke | Actual attention and sparse MoE; finite forward/backward; full optimizer/checkpoint save and resume | One allocated GPU |
| Reference parity | Same-weight outputs and gradients against an external research implementation | One allocated GPU; separate worker processes |

## CPU and native checks

Follow the commands in the [README](../README.md#tests). CPU graph tests instantiate the production `MBTDecoder` with diagnostic blocks, and test the production fusion and shared-KV modules. They do not claim to validate CUDA kernels.

The native smoke constructs a small model using the actual attention, router and expert kernels. It checks causality, trains to update 2, writes a complete native checkpoint, and resumes to update 4. A fresh output directory is required. Paper-size GPU capacity and distributed throughput require their own measurements.

## Optional reference comparison

This check is intended for maintainers who have an external copy of the research implementation. That source is not bundled or imported by the normal model or training entry point. The reference and candidate run in separate processes, so reference-side global modifications cannot affect the explicit model.

```bash
export MBT_REFERENCE_ROOT=/scratch/reference-source
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code python tests/compare_reference.py \
  --candidate-root "$PWD" \
  --output-dir /scratch/tests/parity-unique \
  --temp-root /scratch/tmp
```

The reference root must provide the research `code/mbt` package and its matching `code/backend`. The suite tests Standard, asymmetric Independent-KV, and width-changing 2/3/4-branch Independent-KV and Joint-KV configurations. Both sides use the same native operators and the same weights. The candidate loads the reference weights through an explicit key conversion and verifies the loaded tensors exactly.

The comparison records full-logit error, scalar cross-entropy, every trainable parameter's gradient, and a suffix-perturbation causality check. Missing gradient paths and shape/key mismatches fail. Gradient comparisons use relative L2 error so an absolute tolerance cannot hide a missing small gradient. BF16 causality allows bounded numerical differences: changing suffix routing can change grouped-GEMM shapes even when attention remains causal. All tolerances and measured errors appear in `parity.json`.

Artifacts and logs stay in the requested scratch directory. This is a mechanics comparison, not a new pretraining result or an inference-throughput benchmark.
