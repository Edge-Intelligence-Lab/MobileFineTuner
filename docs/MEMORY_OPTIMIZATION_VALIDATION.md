# Memory feature validation scope

This is a code-integration check, not a paper benchmark, accuracy result, or
claim of bitwise equivalence. Features remain opt-in and disabled by default.

## Tested paths

On macOS arm64 with Apple Clang and Accelerate:

- Release BLAS on and off: 23 actual CTest checks pass. The additional
  `TokenizerHFGolden` entry self-skips without its external fixture; do not count
  that as an executed golden comparison.
- ASan + UBSan, memory pool disabled: four new memory-feature test targets pass.
- Legacy-engine library builds; its attention training fallback is retained.
- Installed-library CMake consumer builds and exercises checkpoint backward.
- Qwen `train_mmlu`, Gemma 270M `train`, and Gemma 1B `train` compile.
- Direct checkpoint + streaming-attention composition checks inferred captures
  and explicit K/V captures; Q, K, and V gradients must all match the unwrapped
  operation. This catches a dependency-capture bug not detected by block tests.

The small two-layer Qwen/Gemma tests cover all eight switch combinations,
GQA, padding, causal/sliding masks, three optimizer updates, and disk reloads.
Gradient comparisons at fixed parameter snapshots are separate from
independently evolving optimizer trajectories. After a floating-point update
changes parameters, later gradients are no longer evaluated at identical inputs.

## Real-weight diagnostic and remaining limitation

Two-update host checks used FP32, batch 1, dropout 0. Qwen2.5-0.5B used a PIQA
MCQ CSV through the existing MMLU-format loader, sequence length 64. Gemma 270M
used PIQA JSONL, sequence length 32. These short, answer-masked checks do not
validate task accuracy, long training stability, or the full dataset pipeline.
No phone, CUDA training, or real-weight Gemma 1B run is claimed here.

Qwen checkpoint-only produced identical 540,672 adapter elements to its paired
dense baseline. The default-off path also matched all 540,672 elements from an
independent build of upstream `086b1a0` on the same two-update workload.
Combined checkpoint/offload/streaming matched streaming-only exactly on this
check; the extra two features did not add adapter differences.
Gemma 270M with all three switches completed two updates and adapter saving;
maximum adapter difference was `1.67036e-7` across 1,898,496 elements.
Enabling streaming changes FP32 reductions compared with GEMM.
In the final diagnostic, the first-step gradient maximum absolute difference was
`9.51346e-6`, but the two-step adapter maximum difference was `6.54949e-5`.
Three of 540,672 coordinates exceeded the initially tested `2e-5` adapter limit.
That strict real-model adapter check **did not pass**; it has not been silently
relaxed or replaced with rounded-loss agreement.

At the worst first-step coordinate (adapter tensor 49, index 5984), both
parameters start at zero. Dense/streaming gradients are `6.2080879e-8` and
`1.1454676e-8`. With unchanged Adam learning rate `2e-4` and epsilon `1e-8`,
saved first/second moments reproduce updates `-1.7225339e-4` and
`-1.0678023e-4`. Thus near-zero-gradient sensitivity explains that update gap;
it is not evidence of disk corruption or a guarantee of unchanged accuracy.
Keep the standard attention path when strict historical adapter reproduction
is required, and validate longer intended workloads before enabling streaming.

No numerical comparison here licenses changing optimizer epsilon, training
settings, or published experimental results to make runs agree. No universal
accuracy or speed/memory-saving claim is made.
