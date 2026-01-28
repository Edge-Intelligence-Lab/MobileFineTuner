/**
 * @file gradient_buffer.cpp
 * @brief 在位梯度累加缓冲实现
 */

#include "gradient_buffer.h"
#include "../core/logger.h"
#include "arena_allocator.h"
#include <cstring>
#include <iostream>

namespace ops {
namespace memory {

InPlaceGradientBuffer::~InPlaceGradientBuffer() {
    // 🔧 使用Arena分配，不需要手动释放
    // StaticWeightArena会在程序结束时统一清理
}

void InPlaceGradientBuffer::initialize(const std::vector<TensorPtr>& params) {
    buffers_.clear();
    buffers_.reserve(params.size());
    total_bytes_ = 0;
    
    #ifdef USE_ARENA_ALLOCATOR
    // 🔧 使用StaticWeightArena分配梯度缓冲（长期持有，不频繁释放）
    auto& static_arena = ArenaManager::instance().static_weights();
    #endif
    
    for (const auto& param : params) {
        size_t size = param->numel();
        size_t bytes = size * sizeof(float);
        
        // 从Arena分配（零初始化在Arena内部完成）
        float* buffer = nullptr;
        #ifdef USE_ARENA_ALLOCATOR
        buffer = static_cast<float*>(static_arena.allocate_static(bytes, "gradient_buffer"));
        if (buffer) {
            std::memset(buffer, 0, bytes);  // Arena不保证零初始化，手动清零
        }
        #else
        buffer = static_cast<float*>(std::malloc(bytes));
        if (buffer) {
            std::memset(buffer, 0, bytes);
        }
        #endif
        
        if (!buffer) {
            throw std::bad_alloc();
        }
        
        buffers_.push_back({buffer, size, true});
        total_bytes_ += bytes;
    }
    
    // 静默初始化（日志太多）
    // OPS_LOG_INFO_F("✅ InPlaceGradientBuffer initialized: %zu buffers, %.2f MB total",
    //                buffers_.size(), total_bytes_ / (1024.0f * 1024.0f));
}

void InPlaceGradientBuffer::accumulate(size_t param_idx, const TensorPtr& grad) {
    if (param_idx >= buffers_.size()) {
        OPS_LOG_ERROR_F("Invalid param_idx: %zu (max: %zu)", param_idx, buffers_.size());
        return;
    }
    
    if (!grad) return;
    
    auto& buf = buffers_[param_idx];
    if (buf.size != static_cast<size_t>(grad->numel())) {
        OPS_LOG_ERROR_F("Gradient size mismatch: expected %zu, got %ld",
                       buf.size, grad->numel());
        return;
    }
    
    // 在位累加: buffer += grad （BLAS-free axpy）
    const float* grad_data = grad->data<float>();
    for (size_t i = 0; i < buf.size; ++i) {
        buf.data[i] += grad_data[i];
    }
}

TensorPtr InPlaceGradientBuffer::get_gradient(size_t param_idx, const std::vector<int64_t>& shape) {
    if (param_idx >= buffers_.size()) {
        return nullptr;
    }
    
    auto& buf = buffers_[param_idx];
    
    // 🔧 使用零拷贝包装模式（NO allocation, NO copy）
    // wrap_external_flag=true 激活外部内存包装
    auto grad_tensor = std::make_shared<Tensor>(shape, buf.data, DType::kFloat32, kCPU, true);
    
    return grad_tensor;
}

void InPlaceGradientBuffer::zero() {
    for (auto& buf : buffers_) {
        if (buf.data) {
            std::memset(buf.data, 0, buf.size * sizeof(float));
        }
    }
}

void InPlaceGradientBuffer::print_stats() const {
    std::cout << "InPlaceGradientBuffer Stats:\n";
    std::cout << "  Num buffers: " << buffers_.size() << "\n";
    std::cout << "  Total size: " << total_bytes_ / (1024 * 1024) << " MB\n";
}

} // namespace memory
} // namespace ops

