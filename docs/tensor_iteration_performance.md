# Tensor iteration optimization

## Scope

- Binary elementwise broadcasting computes aligned source strides once, then
  processes contiguous rows. An outer-coordinate counter replaces per-element
  allocation, division and modulo. Operand order is preserved for subtraction
  and division. Same-shape and unary forward loops cache the element count.
  Broadcasting a zero-sized axis with a singleton now correctly stays empty,
  rather than taking `max(0, 1)` and reading an empty input.
- Transpose and permute materialization copy contiguous trailing blocks when
  possible. Other axes advance with a coordinate counter. Copies use the dtype's
  byte width, including FP16/BF16 and integers; permute previously copied through
  float pointers regardless of dtype. The existing 2-D member-transpose fast path
  is retained. Outputs still own separate dense storage, not views.
- No autograd registration, optimizer, model hyperparameter, or checkpoint format
  is changed. The separate reshape-gradient correction is not included here.

The upstream Qwen output head already calls `matmul_rhs_T(hidden, embed_tokens_)`.
It is deliberately not changed. `QwenHead` adds forward/backward comparison against
explicit transpose + matmul, with frozen/trainable weights, batches 1/2 and
transposed inputs. It uses floating-point tolerances, not a bitwise-equivalence
claim between different matrix-multiplication implementations.

## Reproduce

Run from a complete repository checkout:

```sh
cmake -S operator -B build-cpp -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON -DUSE_BLAS=OFF -DOPERATORS_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -UNDEBUG"
cmake --build build-cpp -j4
ctest --test-dir build-cpp --output-on-failure
ctest --test-dir build-cpp -R 'TensorIteration|QwenHead' -V
```

For macOS Accelerate:

```sh
cmake -S operator -B build-blas -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON -DUSE_BLAS=ON \
  -DCMAKE_CXX_FLAGS="-DACCELERATE_NEW_LAPACK" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -UNDEBUG"
cmake --build build-blas -j4
ctest --test-dir build-blas --output-on-failure
```

On Linux, configure `USE_BLAS=ON` with an installed BLAS implementation and omit
the Accelerate definition. That Linux configuration is not locally validated here.

`TensorIteration` uses independent coordinate-based reference calculations,
covering permutations/transpose, singleton and zero-sized axes, negative axis
indices, seven dtypes, both broadcast operand orders, and addition/multiplication
gradient reductions. Checks remain active in Release builds. Existing gradient
tests supplement it; this is not exhaustive certification of every autograd op.

## Validation boundary

On the tested macOS/AppleClang host, both pure C++ and Accelerate CTest runs report
22/22 successful exits. `TokenizerHFGolden` internally skips without an external
fixture, so these are 21 executed tests plus one fixture-dependent skip, not 22
fully exercised tests. The two new tests execute 1395 copy/broadcast checks and
8 output-head comparisons respectively.

The two new tests also pass with AddressSanitizer/UndefinedBehaviorSanitizer
(memory pool disabled, leak detection disabled). With the legacy autograd engine,
the forward-only `TensorIteration` run passes 1363 checks. The operator library
cross-compiles for Android arm64-v8a/API 24 using NDK 27 with warnings as errors;
no Android execution is claimed.

A fixed-seed, two-layer synthetic Qwen regression separately compared ten optimizer
updates against the unmodified upstream base in both pure C++ and Accelerate builds:
raw FP32 losses, LoRA gradients and post-update parameters were byte-identical
within each backend. This is a small numerical regression,
not full-size Qwen/PIQA accuracy evaluation or proof of cross-engine parity.

Previous end-to-end timing figures from a different local tensor/view implementation
must not be attributed to this dense upstream implementation. No end-to-end speedup,
mobile/Termux result, RSS improvement, or PyTorch/ORT parity is claimed by this PR.

GitHub source archives at the tested base exclude nested `data/` directories due to
an existing export-ignore rule. Use a full checkout, rather than those archives,
to retain the dataset-loader source files needed by CMake.
