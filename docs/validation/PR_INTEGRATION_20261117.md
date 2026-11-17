# PR integration audit — 2026-11-17

Baseline: `be95a289d9cd5ecd93b6fab7fdc3c4551c3f9fb1` (upstream main at audit).
This record covers the candidate reviewed on `validation/macos-20261114`.
It records local acceptance checks; upstream review and hosted CI are separate
from these results.

## Corrections made while integrating

- Unified Qwen2/3 dense and Qwen3.5 hybrid configuration, exact LoRA target
  dispatch, default-target metadata, strict base/adapter validation and PEFT
  save/load. Qwen3.5 default dimensions/RoPE now follow the pinned reference
  configuration. Both text tensor namespaces and native BF16 matrices are tested.
- Fixed GPT-2 fused QKV slicing and partial-column LoRA additions so both fused
  and split-QKV paths retain the Q/K/V adapter gradients, including Q8 bases.
- Fixed FP16 subnormal conversion, large-vocabulary loss-gradient accumulation
  and gradient-norm precision. Invalid/non-finite settings and failed AutoTrainer
  steps now reject and clean up graph/gradients. Zero max norm disables clipping.
- Bound GPT-2/Gemma dropout and seed settings to their adapters; stochastic
  checkpoint recomputation rejects unsupported nonzero dropout. Repeated Auto
  LoRA initialization rejects stale optimizer references.
- Corrected MMLU answer labels to remain aligned with input IDs; the shared loss
  already applies the causal shift. Regression checks the exact gradient positions.
- Unified native matrix storage and target parsing across Qwen CLIs; fixed final
  short gradient-accumulation groups and input validation. Stability CSVs and
  periodic adapter snapshots are available through the WikiText CLI.
- Serialized Android native calls across sessions and synchronized Java instance
  lifecycle methods. Added concurrent synthetic instrumentation coverage.
- Added reproducible HF/PEFT reference generation, a dense/hybrid CI matrix,
  hermetic CLI/controller checks, and a maintained macOS serial stability runner.
  JSON is vendored with its MIT license; runtime training does not use Python.

## Acceptance evidence and its limits

The local review bundle records commands and logs under the ignored
`runs/pr_review_20261117/` directory. Core checks include strict Release builds,
33 CTest tests with supplied tokenizer/reference fixtures, independent tiny
logits/gradient/Adam-update comparisons in both attention modes, PEFT loading of
native exports, full official Qwen3.5 forward/gradient comparison, installed
package consumers, example smoke runs, and Android build artifacts. The final
review report identifies which commands completed successfully.

Local acceptance checks completed:

| Check | Result |
| --- | --- |
| Clean patched source, strict Release + Accelerate | 33/33 CTest passed |
| Strict Release, BLAS disabled | 33/33 CTest passed |
| Tiny HF/PEFT references | 12 configurations × 2 attention modes passed |
| Native exports reloaded/evaluated by PEFT | 24 exports passed |
| Official Qwen3.5 native-storage forward/gradient/SGD check | Both attention modes passed |
| GPT-2 fused/split QKV × FP32/Q8 × ordinary/streamed | 8 gradient/update cases passed, profiling path exercised |
| Installed Operators and MobileFineTuner consumers | Both compiled and ran; public headers/license installed |
| Five example builds and synthetic training | Passed |
| WikiText accumulation/snapshots and MMLU train/resume | Passed, including truncated-answer rejection |
| Detached Mac controller, reports, frozen binary, queued stop | Passed |
| Energy schedule consistency | Exact loss/adapter equality and delays passed |
| Android SDK release AAR, instrumentation APK, local Maven, both apps | Built; device execution pending |
| Android arm64 WikiText and MMLU native CLIs | Built with NDK 26.1 / API 29 |
| Legacy engine library, tests disabled | Strict compile passed; no legacy hybrid-training claim |
| Release tree, workflow syntax, patch completeness | Passed locally; hosted CI has not run |

The official Qwen3.5 native-BF16 comparison uses SGD for the cross-framework
update check. Observed maximum errors were about `6.4e-5` for logits,
`4.85e-4` for gradients and `4.85e-7` for the SGD update. Full-size Adam trajectory
parity is not claimed; see [the historical validation record](QWEN35_20261115.md).
Tiny fixtures independently check Adam and adapter reload.

The earlier long-run queue used the pre-audit executable: its Qwen3.5 300-step
run and 30-step adapter resume completed, but that version had less precise
clipping and mixed linear-mixer storage. These results are preliminary. The final
candidate queue freezes a new executable and repeats the three official models
with 300 steps plus 30 resume steps each and a Qwen3.5 length-128 stress check.
Its live/final report is a separate runtime artifact; pending jobs are not passes.
See [Mac stability instructions](../MAC_STABILITY.md).

AAR/APK/native cross-compilation does not execute Android instrumentation.
Physical-device model training, memory/thermal behavior and lifecycle tests
remain pending a connected phone. No blanket production-readiness claim follows
from these checks. Supported training is pure text; Qwen3.5 vision/MoE/cache,
checkpointing and offload remain outside the supported contract.
