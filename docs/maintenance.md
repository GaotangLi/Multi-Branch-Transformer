# Maintaining a release

Keep the public training interface in `code/mbt/`, experiment recipes in
`configs/`, and native backend changes explicit in `code/backend/PORTABILITY.md`.
Avoid importing cluster-specific launch policies into model code.

Keep architecture changes in the explicit MBT module hierarchy. Forward routing
must remain visible in `MBTModel` and `MBTDecoder`; do not introduce global case
selection, routing hooks, or backend forward replacement. Validation and
checkpoint behavior belong in the explicit training lifecycle. The historical
reference implementation used for parity checks stays outside the public tree.

Before cutting a release:

1. Run CPU configuration/launcher tests and numerical decoder output/gradient
   checks in a clean environment with CPU PyTorch.
2. Run the native single-GPU smoke on a supported GPU and retain its summary.
3. Check full training save/resume with optimizer and data progress when runtime
   or checkpoint code changes. Keep the `mbt-hierarchical-v2` checkpoint and run
   manifest contract explicit; raw weight-key conversion does not establish
   optimizer-resume compatibility. Capacity is tested at the intended EP layout.
4. Confirm README commands and all local links, with no private paths,
   credentials, datasets, checkpoints, or generated caches in the Git diff.
5. Update the validation record and dependency snapshot. Tag a concrete Git
   revision; do not label an untested environment or model layout as validated.

CI runs CPU tests only. GPU execution belongs to an explicitly allocated
runner; pull requests must never gain unattended access to cluster credentials.
