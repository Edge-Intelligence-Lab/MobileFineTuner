/**
 * @file gradient_buffer.h
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {
namespace memory {

class InPlaceGradientBuffer {
private:
    struct GradBuffer {
        float* data = nullptr;
        size_t size = 0;
        bool owned = true;
    };
    std::vector<GradBuffer> buffers_;
    size_t total_bytes_ = 0;

public:
    InPlaceGradientBuffer() = default;
    ~InPlaceGradientBuffer();

    void initialize(const std::vector<TensorPtr>& params);
    void accumulate(size_t param_idx, const TensorPtr& grad);
    TensorPtr get_gradient(size_t param_idx, const std::vector<int64_t>& shape);
    void zero();

    size_t total_bytes() const { return total_bytes_; }
    void print_stats() const;

    InPlaceGradientBuffer(const InPlaceGradientBuffer&) = delete;
    InPlaceGradientBuffer& operator=(const InPlaceGradientBuffer&) = delete;
};

} // namespace memory
} // namespace ops


