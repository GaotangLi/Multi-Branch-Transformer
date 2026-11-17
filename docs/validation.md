# Validation record

Explicit MBT refactor, 2026-11-07:

| Check | Status |
|:---|:---|
| CPU decoder graph, output/gradient, fusion and Joint-KV tests | Passed |
| CPU recipe, launcher, resume-identity, validation-state and retention tests | Passed; 20 CPU tests in total |
| All 44 W1–W4 recipes, native argument parsing and model configuration validation | Passed on CPU |
| Import explicit MBT model, native attention/MoE dependencies and pretraining entry | Passed in the recorded environment |
| Build a wheel and import its packaged backend/model from a separate installation | Passed |
| Parse source, check documentation links and scan for private identifiers | Passed |
| Same-weight native outputs/gradients against the preceding implementation | Awaiting one-GPU allocation |
| Native forward/backward, causality and full-state save/resume | Awaiting one-GPU allocation |
| Refactored launcher's full EP8 training run | Not yet run |

The CPU decoder tests exercise the actual `MBTDecoder` with diagnostic blocks,
including unequal branch depths, absent outer stacks, width changes,
cross-branch gradients and synchronized Joint-KV inputs. Native parser checks
substitute an SM100 capability response on CPU; they do not exercise a GPU.

The one-GPU suite compares identical serialized weights, then checks native
training and strict optimizer/checkpoint resume. See [testing.md](testing.md)
for commands, tolerances and coverage. Initialization scales are preserved, but
fresh random draws can differ because unused parameters are no longer built.

The native attention, MoE and training components originate from the research
implementation. Historical training results and CPU checks do not substitute
for GPU qualification of the new composition. Run the native checks before a
long experiment in a new environment.
