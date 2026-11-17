# Third-party notices

The root BSD-3-Clause license covers this project's original contributions,
documentation, and original figures. It does not replace upstream notices.

| Component | Scope | License / source |
|:---|:---|:---|
| Modified Megatron-LM training backend | `code/backend/` | Full original [license bundle](code/backend/LICENSE), including NVIDIA BSD terms and the licenses of incorporated code; file headers remain intact. Upstream: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM). |
| Native MBT model adapters | `code/mbt/_native/` | Adapted from this project's research implementation on the Megatron backend; retain the applicable backend notices in addition to the root license. |
| PyTorch, Transformer Engine, FlashAttention, Apex | Installed dependencies, not vendored binaries | Refer to each dependency's distribution and license. |
| Model weights and tokenizer | Supplied separately | The corresponding model card and tokenizer license apply. |
| Training dataset | Supplied / preprocessed separately | The selected dataset's card and upstream terms apply. |

`code/backend/SOURCE_MANIFEST.json` records the exact input source revision and
hashes. Portable build/runtime changes are documented in
`code/backend/PORTABILITY.md`; MBT-specific model adaptations are described in
[provenance](docs/provenance.md). No dataset, model weights, proprietary font
files, or third-party compiled extensions are bundled in the repository.
