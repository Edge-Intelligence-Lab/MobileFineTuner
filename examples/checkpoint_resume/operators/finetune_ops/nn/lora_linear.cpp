/**
 * @file lora_linear.cpp
 * @brief LoRA-enhanced linear layer implementation
 */

#include "lora_linear.h"
#include "../core/ops.h"
#include <stdexcept>

namespace ops {

const TensorPtr& LoRALinear::base_W() const {
    static TensorPtr null_tensor;
    return (W_ref_ && *W_ref_) ? *W_ref_ : null_tensor;
}

const TensorPtr& LoRALinear::base_b() const {
    if (b_ref_ && *b_ref_) return *b_ref_;
    return b_cache_;
}

void LoRALinear::attach_lora(const TensorPtr& A, const TensorPtr& B,
                             float scale, int col0, int cols) {
    if (!A || !B) {
        throw std::runtime_error("LoRALinear::attach_lora: A and B must not be null");
    }
    
    // Shape validation
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("LoRALinear::attach_lora: A and B must be 2D");
    }
    
    int64_t out_cols = (cols <= 0) ? B->shape()[1] : cols;
    
    // LoRA parameters require gradients
    A->set_requires_grad(true);
    B->set_requires_grad(true);
    
    slices_.emplace_back(A, B, scale, col0, out_cols);
}

void LoRALinear::clear_lora() {
    slices_.clear();
    merged_ = false;
}

TensorPtr LoRALinear::forward(const TensorPtr& x) const {
    auto W = base_W();
    auto b = base_b();
    if (!W) throw std::runtime_error("LoRALinear::forward: base weight is null (sharded and not loaded?)");

    // Base branch: x @ W + b (base is frozen, no gradients needed)
    auto y = matmul(x, W);
    if (b) {
        y = add(y, b);
    }
    
    // LoRA delta: Σ scale * (x @ A @ B)
    // If already merged, don't compute LoRA dynamically (already in base)
    if (!slices_.empty() && !merged_) {
        int64_t out_dim = W->shape()[1];
        
        for (const auto& slice : slices_) {
            // Ensure A and B require gradients
            if (!slice.A->requires_grad()) {
                const_cast<Tensor*>(slice.A.get())->set_requires_grad(true);
            }
            if (!slice.B->requires_grad()) {
                const_cast<Tensor*>(slice.B.get())->set_requires_grad(true);
            }
            
            // x @ A
            auto xA = matmul(x, slice.A);  // [B, S, rank]
            
            // (x @ A) @ B
            auto xAB = matmul(xA, slice.B);  // [B, S, out_slice]
            
            // Scale
            auto delta = mul(xAB, slice.scale);
            
            // If slice.cols == out_dim, directly add; otherwise need partial column update
            if (slice.cols == out_dim && slice.col0 == 0) {
                y = add(y, delta);
            } else {
                // Partial column update: manual operation needed
                // y[:, :, col0:col0+cols] += delta
                float* y_data = y->data<float>();
                const float* delta_data = delta->data<float>();
                
                auto y_shape = y->shape();
                int64_t B = y_shape[0];
                int64_t S = y_shape[1];
                
                for (int64_t b = 0; b < B; ++b) {
                    for (int64_t s = 0; s < S; ++s) {
                        for (int64_t c = 0; c < slice.cols; ++c) {
                            int64_t y_idx = b * S * out_dim + s * out_dim + (slice.col0 + c);
                            int64_t delta_idx = b * S * slice.cols + s * slice.cols + c;
                            y_data[y_idx] += delta_data[delta_idx];
                        }
                    }
                }
            }
        }
    }
    
    return y;
}

void LoRALinear::merge_to_base() {
    if (merged_ || slices_.empty()) return;
    
    auto W = base_W();
    if (!W) throw std::runtime_error("LoRALinear::merge_to_base: base weight is null");

    // Compute ΔW of all LoRA slices and add to base
    for (const auto& slice : slices_) {
        // ΔW_slice = A @ B * scale  -> [in, out_slice]
        auto delta_W = matmul(slice.A, slice.B);
        delta_W = mul(delta_W, slice.scale);
        
        // Add in-place to sub-matrix W[:, col0:col0+cols]
        // Simplified implementation: add directly to entire W (assuming slice.col0=0, cols=full columns)
        // TODO: Full implementation requires sub-matrix operations
        float* W_data = W->data<float>();
        const float* delta_data = delta_W->data<float>();
        
        int64_t in_dim = W->shape()[0];
        int64_t out_dim = W->shape()[1];
        
        // Verification: for non-split case, delta should match W shape
        if (delta_W->shape()[0] == in_dim && delta_W->shape()[1] == slice.cols) {
            // Add to specified column range
            for (int64_t i = 0; i < in_dim; ++i) {
                for (int64_t j = 0; j < slice.cols; ++j) {
                    int64_t W_idx = i * out_dim + (slice.col0 + j);
                    int64_t delta_idx = i * slice.cols + j;
                    W_data[W_idx] += delta_data[delta_idx];
                }
            }
        } else {
            throw std::runtime_error("LoRALinear::merge_to_base: shape mismatch");
        }
    }
    
    merged_ = true;
}

void LoRALinear::unmerge_from_base() {
    if (!merged_ || slices_.empty()) return;
    
    auto W = base_W();
    if (!W) throw std::runtime_error("LoRALinear::unmerge_from_base: base weight is null");

    // Subtract LoRA delta from base
    for (const auto& slice : slices_) {
        auto delta_W = matmul(slice.A, slice.B);
        delta_W = mul(delta_W, slice.scale);
        
        float* W_data = W->data<float>();
        const float* delta_data = delta_W->data<float>();
        
        int64_t in_dim = W->shape()[0];
        int64_t out_dim = W->shape()[1];
        
        if (delta_W->shape()[0] == in_dim && delta_W->shape()[1] == slice.cols) {
            for (int64_t i = 0; i < in_dim; ++i) {
                for (int64_t j = 0; j < slice.cols; ++j) {
                    int64_t W_idx = i * out_dim + (slice.col0 + j);
                    int64_t delta_idx = i * slice.cols + j;
                    W_data[W_idx] -= delta_data[delta_idx];
                }
            }
        }
    }
    
    merged_ = false;
}

std::vector<TensorPtr> LoRALinear::trainable_parameters() const {
    std::vector<TensorPtr> params;
    params.reserve(slices_.size() * 2);
    for (const auto& slice : slices_) {
        params.push_back(slice.A);
        params.push_back(slice.B);
    }
    return params;
}

std::vector<std::pair<std::string, TensorPtr>> LoRALinear::debug_params() const {
    std::vector<std::pair<std::string, TensorPtr>> result;
    for (size_t i = 0; i < slices_.size(); ++i) {
        std::string prefix = debug_name_.empty() ? ("lora_" + std::to_string(i)) : debug_name_;
        result.emplace_back(prefix + "_lora_A_default_weight", slices_[i].A);
        result.emplace_back(prefix + "_lora_B_default_weight", slices_[i].B);
    }
    return result;
}

}  // namespace ops
