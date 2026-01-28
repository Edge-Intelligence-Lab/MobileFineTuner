/**
 * @file simple_tensor_stub.cpp
 * @brief Simple stub implementsation for Tensor::numel() to avoid complex dependencies
 */

#include "../core/tensor.h"
#include <numeric>

namespace ops {

// Simple stub implementsation of numel() method
int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

} // namespace ops
