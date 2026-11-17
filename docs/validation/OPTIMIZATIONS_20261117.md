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
| Tokenizers | Three external HF fixture sets execute 16, 28 and 9 records respectively; passed. The mixed sets cover GPT-2 small, Gemma 270M and Qwen dense tokenizers; the third covers Qwen3.5. Sets overlap and these are not counts of unique prompts or all seven model tokenizers. |

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

## Completed real-checkpoint and stability checks — 2026-11-18

These follow-up checks use binaries pinned from the `d9ad0a5` code snapshot
(the subsequent `216b29c` commit changes documentation only). The default long
run uses an earlier pinned binary; its scope is separated below. On the tested
code revision, hosted Mac build/test and Linux dense/hybrid reference CI all
completed successfully. Later documentation-only commits may trigger new CI.

### Official Qwen3.5 full-weight optimization matrix

All eight combinations of ordinary/streamed attention, checkpointing off/on and
frozen-weight offload off/on pass independent HF logits, every adapter gradient,
one SGD update and post-update output checks. Non-offloaded profiles use native
base storage; offload profiles promote the same checkpoint values to FP32.
Maximum observed logit error is `6.38962e-5`, adapter-gradient absolute error
`4.84467e-4`, and one-SGD-update error `4.84288e-7`. The full-model gradient check
also enforces per-tensor relative-L2 and scale-relative maximum bounds; absolute
maxima alone are not its acceptance rule. Full-size Adam trajectory parity is
not claimed.

Each offload profile registers all 320 frozen tensors, writes 3,009,572,096
payload bytes and reloads them during execution. Successful read payload is
12,080,072,448 bytes without checkpointing and 14,072,521,728 with checkpointing.
These count tensor payload operations, not physical uncached disk traffic.
The serialized eight-profile job completes in 131.30 seconds with sampled
peak RSS 3867.47 MiB. This is a short numerical check, not an optimized long run.

Earlier concurrent attempts stopped on global swap-growth protection while
moving to FP32 offload. Their logs and guard failures remain preserved; no pass
was assigned to those interrupted attempts. Deferred allocation and direct
transpose reduced extra loading allocations. After the long queue released its
resources, the serialized run completed with the same guard policy. No user
applications or OS swap settings were changed to force completion.

### Latest Gemma trainer with real weights

Gemma 3 270M and 1B-PT each pass three optimizer updates in ordinary mode and
combined checkpointing + streamed attention + FP32 offload mode. Sequence length
is 16, batch/accumulation 1, rank 2, alpha 4, dropout 0, LR 2e-4, no warmup.

| Model/path | Logged losses | Adapter tensors finite and changed from step 1 to 3 | Sampled peak RSS MiB |
| --- | --- | ---: | ---: |
| 270M ordinary | 6.75599, 9.061, 5.603 | 252/252 | 3235.16 |
| 270M combined | 6.75599, 9.061, 5.603 | 252/252 | 3269.00 |
| 1B-PT ordinary | 6.27291, 7.817, 5.218 | 364/364 | 4412.69 |
| 1B-PT combined | 6.27292, 7.817, 5.218 | 364/364 | 4531.27 |

Every B matrix is nonzero and every final export equals its step-3 snapshot.
Combined runs report positive payload reads/writes and loads/evictions: 270M
reads 6,434,348,544 and writes 1,743,481,344 bytes; 1B reads 23,997,249,024 and
writes 5,207,503,360 bytes. Logged losses use limited display precision; adapter
values and I/O checks provide additional acceptance evidence. Varying minibatches
mean these three losses are not a convergence benchmark. These results validate
the new trainer on full checkpoints, without claiming Gemma long-run stability.

### Default-path Qwen stability queue

The queue completed at **2026-11-18 01:06:29 CST**. All three models pass 300
sequence-32 optimizer updates and 30 adapter-resume updates with a fresh
optimizer. All losses, gradient metrics and adapter tensors are finite; all
adapter tensors change on resume. The runner flags neither sustained RSS growth
nor significant slowdown on the covered long runs.

| Model | Long + adapter-resume steps | Sampled long-run peak RSS MiB | Late-minus-early median RSS MiB | Late/early median step time |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 300 + 30 | 2934.30 | -1.55 | 0.996 |
| Qwen3-0.6B | 300 + 30 | 2042.14 | +0.42 | 0.998 |
| Qwen2.5-0.5B | 300 + 30 | 1634.11 | -4.39 | 1.014 |

Qwen3.5 additionally passes five sequence-128 stress updates from the resumed
adapter, with sampled peak RSS 6862.11 MiB. This short stress result does not
qualify arbitrary longer sequences or every device memory limit.

The long queue pins pre-refactor executable SHA-256
`fde3c388ba26f06de68f13ff5831e46061a499d448927888ead4fbfc05fd54cb`.
A separate three-step Qwen3.5 default run using the new pinned CLI reproduces
every adapter value of the old binary's calibration run exactly. This bridges
the covered default-path short comparison; it does not turn the old binary's
long run into a 300-step run of new optimized paths. The new optimized paths
have the separate short real-weight matrix above. Optimized long-duration
training and Gemma long runs remain outside completed qualification.

Raw evidence is under `runs/pr_candidate_soak_20261117_retry/` and
`runs/optimization_review_20261117/real_matrix_after_soak/`. The completion check
re-reads status files, all eight HF results, frozen-binary hashes, exact default
adapter values, Gemma tensors/snapshots and positive offload payload statistics.

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
