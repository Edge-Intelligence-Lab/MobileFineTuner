#pragma once

#include "tensor.h"

namespace ops {

// CPU FP32, first-order, deterministic activation recomputation (new engine only).
// fn must use tracked ops, be side-effect free, and have the same dependencies on
// replay. No RNG/dropout state is saved. Do not mutate input/captured tensors until
// backward completes. Captured differentiable tensors may be leaves OR non-leaves.
// Frozen weights may be reloaded by fn; their values must remain identical.
// Pass trainable parameters/captures explicitly for model integration. Tracked
// boundary dependencies are also discovered and deduplicated automatically.
// The existing singleton engine is not safe for concurrent training calls.
TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input,
                     const std::vector<TensorPtr>& captured_tensors = {});

} // namespace ops
