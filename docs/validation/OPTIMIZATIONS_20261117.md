# Architecture and optimization integration — 2026-11-17

This follow-up records the execution-contract changes in PR #12. It separates
implemented functionality, small-model numerical validation, real checkpoints,
long runs and physical-device execution. The current product scope is CPU text
LoRA, with optional BLAS. See the [model/optimization matrix](../MODEL_SUPPORT.md).

## Architecture and correctness changes

- One engine-aware registry declares supported text architectures and training
  capabilities. Unsupported Gemma/Qwen variants cannot silently select a similar
  decoder. Model support does not follow arbitrary checkpoint-name substrings.
- Qwen's stable module inventory drives LoRA, adapter names, checkpoint captures
  and offloaded base-weight leases. Its complete weight inventory drives strict
  loading and deterministic offload grouping, including every Qwen3.5 DeltaNet
  convolution, decay/bias, gated norm, projection, MLP and output-head weight.
- Both Qwen dataset frontends share execution parsing/validation/application.
  The model owns output-head residency for streamed loss; CLIs do not dereference
  evicted head storage. Qwen3.5 now implements block checkpointing and frozen
  FP32 disk offload, including combinations with streamed attention/loss.
- Deferred Qwen pretrained construction allocates shape metadata before loading,
  validates the complete checkpoint before installation, and rejects training or
  adapter initialization while pending. SafeTensors conversion and transpose
  write directly into the final tensor instead of an extra full transpose copy.
- Common evaluation guards restore the previous autograd state and suppress
  adapter dropout during evaluation. Gemma uses shared precise gradient clipping,
  validates training settings, flushes accumulation tails by their actual count,
  and clears failed-step state. Its LR horizon counts per-epoch tail updates.

[Execution architecture source and preview](../../figures/mf-execution-contracts.md).

## Small-model acceptance

Tests use strict Release builds on Apple M2 / 16 GiB. Runtime fixtures, datasets,
weights, adapters, resource samples and raw logs remain outside the source tree,
under `runs/optimization_review_20261117/` and `runs/pr_review_20261117/`.

| Check | Scope and result |
| --- | --- |
| CTest | 34 entries pass with BLAS on and off. External tokenizer comparisons are additionally supplied and run explicitly. |
| HF numerical references | 12 tiny dense/hybrid fixtures × 8 memory/attention profiles = 96 comparisons of logits, every adapter gradient, optimizer updates and post-update logits; passed. Includes tied/untied heads, selected/all targets, native BF16 matrices, both hybrid weight namespaces and longer tiny sequences. |
| Memory composition | Qwen2, Qwen3, Gemma and three Qwen3.5 variants × 8 switches; all adapter gradients at identical parameter snapshots and separate three-Adam-update trajectories pass. Additional streamed-head and offload ownership cases pass. |
| Actual offload | Full frozen-weight inventory count, successful payload I/O, reload/eviction and adapter export checked; flag acceptance alone is insufficient. |
| Gemma trainer | Three microbatches with accumulation=2 compared to two independently calculated Adam updates; clipped/unclipped cases, invalid-label recovery and deterministic gradient-free evaluation pass. |
| GPT-2 QKV | All eight FP32/Q8 × fused/split QKV × ordinary/streamed cases validate connected Q/K/V gradients and parameter updates. |
| Qwen CLI | Tiny Qwen2/Qwen3.5 each run 16 memory/head combinations, four native-storage modes, energy exactness, combined-mode adapter resume and invalid-option rejection. WikiText accumulation-tail and MMLU tests pass. |
| PEFT | 24 native standard/streamed exports loaded and evaluated with upstream PEFT; passed. |
| Tokenizers | 16 existing five-entrypoint cases, 28 Qwen dense cases and 9 Qwen3.5 cases supplied as external HF golden fixtures; passed. These counts include distinct fixture records, not a claim of unique prompts. |

Tiny memory-composition comparisons use elementwise gradient `atol=2e-5,
rtol=1e-4` and trajectory differences below `2e-5`. Direct attention tests also
use finite differences. Checkpointing lowers retained graph-node counts in the
covered examples (e.g. 213→37 and 323→54). This is neither a peak-RSS benchmark
nor evidence of a universal memory saving or speedup.

## Packaging and integration acceptance

- All five example builds and synthetic two-step training smoke runs pass from
  an isolated checkout populated with the reviewed source.
- Both installed `Operators` and `MobileFineTuner` CMake consumers compile and run.
- Android arm64 `train_wikitext` and `train_mmlu` cross-compile; SDK release AAR,
  instrumentation APK and app debug APK build successfully. No device execution
  is inferred from a successful package build.
- Detached-controller pinning, training/resume/reporting and queued-stop checks,
  energy schedule exactness, CLI invalid-option checks and release-tree audit pass.
- A legacy-engine library build succeeds. The capability registry keeps its
  unsupported new-engine paths disabled.

## Real checkpoints and outstanding qualification

The current official Qwen3.5-0.8B reference check passed native base storage with
ordinary/streamed attention, each with checkpointing off/on: four profiles.
Maximum observed logit error is `6.31809e-5`, adapter-gradient absolute error
`4.84467e-4`, and one-SGD-update error `4.84288e-7`. The full-model gradient check
also enforces per-tensor relative-L2 and scale-relative maximum bounds; absolute
maxima alone are not its acceptance rule. This is not full-size Adam trajectory
parity.

The attempted eight-profile real-weight matrix stopped on configured global
swap-growth protection while moving to FP32 offload. No numerical pass is
assigned to the uncompleted profiles. Three attempts preserve their logs;
constructor allocation and transpose copies were reduced between attempts.
The final attempt stopped after 130.31 s, with sampled RSS peak 2894.33 MiB;
global swap reached 11132.12 MiB against a 10789.81 MiB stop threshold. The guard
was retained and no user applications or OS swap settings were changed.

Real-model offload profiles, the new-default versus pinned-long-run adapter
comparison, and updated Gemma 270M/1B ordinary/combined-mode three-step checks are
queued serially after the current long-run queue. They are **pending**, not
passed. Prior real Gemma regression results refer to the code revision named in
[that record](GEMMA_20261117.md); the new trainer changes do not inherit full-size
qualification merely from those older runs.

The active long queue uses a pinned pre-refactor executable. Earlier preliminary
Qwen runs completed 300 + 30 adapter-resume steps per model; their sequence-128
stress run stopped on its resource guard. The active final-candidate run and the
new optimized paths do not inherit a completed long-run result from those runs.
Check the controller status/report rather than assuming elapsed time implies
completion.

A prior real-model streaming diagnostic exceeded its strict `2e-5` adapter bound
at three of 540,672 coordinates; its failure remains recorded in
[the earlier memory validation](../MEMORY_OPTIMIZATION_VALIDATION.md). Changed
FP32 reductions and Adam near-zero-gradient sensitivity mean streaming is not
claimed bitwise or universally trajectory-equivalent to the dense path. No
optimizer setting or historical threshold was changed to turn that into a pass.

Physical-phone training, long-duration Gemma runs, every checkpoint size and
configuration, and task accuracy are not qualified by these host checks.

## Reproduce the focused checks

```bash
cmake -S operator -B runs/build -DCMAKE_BUILD_TYPE=Release \
  -DOPERATORS_WARNINGS_AS_ERRORS=ON -DUSE_NEW_AUTOGRAD_ENGINE=ON -DUSE_BLAS=ON
cmake --build runs/build -j2
ctest --test-dir runs/build --output-on-failure

# Generate small dense/hybrid references in compatible pinned Python environments
# as described in QWEN_DENSE_SUPPORT.md, then run each fixture root:
MFT_QWEN_HF_FIXTURES="$HF_FIXTURES" MFT_QWEN_MEMORY_MATRIX=1 \
  ctest --test-dir runs/build -R QwenHFParity --output-on-failure

cmake -S examples/qwen_lora_finetune -B runs/qwen-cli -DCMAKE_BUILD_TYPE=Release
cmake --build runs/qwen-cli -j2
python3 scripts/test_qwen_training_cli.py \
  --binary runs/qwen-cli/train_wikitext --mmlu-binary runs/qwen-cli/train_mmlu
python3 scripts/test_qwen_execution_options.py \
  --binary runs/qwen-cli/train_wikitext --qwen35-base "$HF_HYBRID_FIXTURES/qwen35_native/base"
```

Repeat the Release build with `-DUSE_BLAS=OFF`. A legacy-engine library build is
also checked, but its capability declaration excludes the new-engine memory
paths and Qwen3.5 LoRA. Runtime training stays in native C++; Python is used for
independent references and test orchestration only.
