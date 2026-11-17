# Contributing

Contributions to model correctness, training portability, documentation, and new MBT experiments are welcome.

## Before opening a pull request

1. Keep the change focused. For a new model variant, state its block allocation, dimensions, fusion, and KV-sharing behavior.
2. Run the CPU tests and the relevant model checks described in the [README](README.md#tests). Report which hardware and checks you actually used.
3. Add a regression test for a corrected numerical or checkpoint behavior. Use small synthetic inputs where possible.
4. Update the relevant configuration example or documentation when the interface changes.

Architecture changes should report both physical block count and depth. Compute comparisons should identify the token budget and distinguish body compute from the embedding/LM-head contribution. Throughput claims should specify the GPU, precision, context length, batch or concurrency, and measurement procedure.

Keep the MBT graph explicit in `MBTModel`, `MBTDecoder`, and their child modules. Pass shared KV and training lifecycle state through arguments or owned modules. Do not implement new variants by replacing backend forwards, registering routing hooks, or selecting a process-global experiment case. Cover graph changes with numerical CPU tests and, when native operators change, the relevant GPU checks.

## Reproducible bug reports

Include the command, configuration, expected behavior, observed error, and environment versions. For distributed problems, include world size and tensor/expert/pipeline parallel sizes. A short log excerpt and the smallest reproduction are more useful than a complete training log.

Remove access tokens, private dataset paths, and personal information before posting. Keep model weights, datasets, runtime caches, and training outputs out of pull requests.

## Code and attribution

Changes to the original project code follow the repository [license](LICENSE). Preserve upstream copyright and license notices for bundled code, and describe any changes to third-party sources in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
