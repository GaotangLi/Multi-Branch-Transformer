# Portable backend changes

Input source revision and file hashes are recorded in `SOURCE_MANIFEST.json`.
Original copyright headers and the complete upstream license are preserved.

- Package the native dependency as `mbt_backend`. The MBT factory loads this
  bundled backend and rejects an already imported, different Megatron build.
- Build dataset C++ helpers under explicit `MBT_DATASET_HELPER_CACHE`, keyed by
  source and ABI, with a file lock and atomic rename. Nothing is compiled into
  the source checkout.
- Remove the legacy root GPT training entry points. The public entry is
  `mbt.train`, with its own explicit model and dataset providers.
- Add an optional `lifecycle` argument to native pretraining. It receives
  model/loader readiness, validation, and completed checkpoint callbacks. The
  default native behavior is unchanged when no lifecycle is supplied.
- Make the optional multi-token-prediction checkpoint helper a lazy import in
  `LanguageModule`; MBT's embedding, head and loss helpers do not need it.

The MBT computation graph is in `code/mbt`, not in backend forward hooks or
class replacements. Generic backend modules retained for transitive imports
are dependencies, not additional supported experiment families. Public model
configuration supports Standard and MBT only.
