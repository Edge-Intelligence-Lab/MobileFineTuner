/**
 * @file gradient_buffer.h
 * @brief 在位梯度累加缓冲 - 避免 clone 导致的内存泄漏
 * 
 * 核心思路：
 * - 为每个可训参数预分配固定大小的梯度缓冲
 * - 累加时直接在位 axpy，不创建临时对象
 * - 更新后就地清零，无需 reset/clone
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
    
    // 初始化：为每个参数预分配梯度缓冲
    void initialize(const std::vector<TensorPtr>& params);
    
    // 累加梯度（在位 axpy: buffer += grad）
    void accumulate(size_t param_idx, const TensorPtr& grad);
    
    // 获取累积的梯度（无拷贝，返回指针包装的 Tensor）
    TensorPtr get_gradient(size_t param_idx, const std::vector<int64_t>& shape);
    
    // 清零所有梯度缓冲
    void zero();
    
    // 统计
    size_t total_bytes() const { return total_bytes_; }
    void print_stats() const;
    
    // 禁止拷贝
    InPlaceGradientBuffer(const InPlaceGradientBuffer&) = delete;
    InPlaceGradientBuffer& operator=(const InPlaceGradientBuffer&) = delete;
};

} // namespace memory
} // namespace ops

