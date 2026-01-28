/**
 * @file memory_efficient_attention.cpp
 * @brief Memory-efficient attention implementation
 */

#include "memory_efficient_attention.h"
#include "ops.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
) {
    // First pass: compute normalization denominator (use max_val for stability)
    double sum_exp = 0.0;
    for (int64_t j = 0; j < seq_len; ++j) {
        sum_exp += std::exp(logits[j] - max_val);
    }
    
    // Second pass: compute normalized weights and accumulate into output
    float inv_sum = 1.0f / static_cast<float>(sum_exp);
    for (int64_t j = 0; j < seq_len; ++j) {
        float weight = std::exp(logits[j] - max_val) * inv_sum;
        const float* v_row = values + j * head_dim;
        
        for (int64_t d = 0; d < head_dim; ++d) {
            output[d] += weight * v_row[d];
        }
    }
}

TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask,
    const MemoryEfficientAttentionConfig& config
) {
    // Validate input shapes
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();
    
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error("memory_efficient_attention: inputs must be 4D [B,H,S,D]");
    }
    
    int64_t batch = q_shape[0];
    int64_t n_head = q_shape[1];
    int64_t seq_len = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Validate k/v shapes match
    if (k_shape[0] != batch || k_shape[1] != n_head || k_shape[2] != seq_len || k_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: k shape mismatch");
    }
    if (v_shape[0] != batch || v_shape[1] != n_head || v_shape[2] != seq_len || v_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: v shape mismatch");
    }
    
    // Auto-compute scale factor
    float scale = (config.scale > 0) ? config.scale : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    
    // Prepare causal mask (if provided)
    const float* mask_data = nullptr;
    if (causal_mask) {
        if (causal_mask->shape().size() != 2 || 
            causal_mask->shape()[0] != seq_len || 
            causal_mask->shape()[1] != seq_len) {
            throw std::runtime_error("memory_efficient_attention: causal_mask must be [S,S]");
        }
        mask_data = causal_mask->data<float>();
    }
    
    // Create output tensor
    auto context = zeros({batch, n_head, seq_len, head_dim}, kFloat32, kCPU);
    
    const float* q_data = q->data<float>();
    const float* k_data = k->data<float>();
    const float* v_data = v->data<float>();
    float* ctx_data = context->data<float>();
    
    // Main loop: iterate over batch, head, and query rows
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            // Base offset for current head
            int64_t head_offset = (b * n_head + h) * seq_len * head_dim;
            const float* q_head = q_data + head_offset;
            const float* k_head = k_data + head_offset;
            const float* v_head = v_data + head_offset;
            float* ctx_head = ctx_data + head_offset;
            
            // Process each query position
            for (int64_t i = 0; i < seq_len; ++i) {
                const float* q_row = q_head + i * head_dim;
                float* ctx_row = ctx_head + i * head_dim;
                
                // === Pass 1: compute scores and max (numerically stable) ===
                // Reuse per-row buffer to avoid repeated allocations
                static thread_local std::vector<float> scores_buf;
                if (scores_buf.size() < static_cast<size_t>(seq_len)) {
                    scores_buf.resize(seq_len);
                }
                float* scores = scores_buf.data();
                float max_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t j = 0; j < seq_len; ++j) {
                    // Compute dot product: q[i] · k[j]
                    const float* k_row = k_head + j * head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    
                    // Apply scale
                    float score = dot * scale;
                    
                    // Apply causal mask
                    if (config.use_causal_mask && j > i) {
                        score = -1e10f;  // Mask upper triangle
                    }
                    
                    // Apply extra mask (if provided)
                    if (mask_data) {
                        score += mask_data[i * seq_len + j];
                    }
                    
                    scores[j] = score; // Write into reusable buffer
                    max_score = std::max(max_score, score);
                }
                
                // === Pass 2: online softmax and accumulate into context ===
                // Initialize output row
                std::fill(ctx_row, ctx_row + head_dim, 0.0f);
                
                // Compute normalization denominator
                double sum_exp = 0.0;
                for (int64_t j = 0; j < seq_len; ++j) {
                    sum_exp += std::exp(scores[j] - max_score);
                }
                
                // Compute weighted sum
                float inv_sum = 1.0f / static_cast<float>(sum_exp);
                for (int64_t j = 0; j < seq_len; ++j) {
                    float weight = std::exp(scores[j] - max_score) * inv_sum;
                    const float* v_row = v_head + j * head_dim;
                    
                    for (int64_t d = 0; d < head_dim; ++d) {
                        ctx_row[d] += weight * v_row[d];
                    }
                }
            }
        }
    }
    
    // Enable gradient propagation if needed
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        context->set_requires_grad(true);
        
        // TODO: implement backward for memory-efficient attention
        // Current strategy: fall back to standard attention or manual grads when training
        // Full implementation needs max_score/sum_exp saved or recomputed in backward
        context->set_grad_fn([q, k, v, causal_mask, scale](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Placeholder: notify user that full backward is needed
            std::cerr << "[WARN] memory_efficient_attention backward not fully implemented yet" << std::endl;
            
            // Return zero grads for now (implement full backward or switch to standard attention when needed)
            auto grad_q = zeros(q->shape(), q->dtype(), q->device());
            auto grad_k = zeros(k->shape(), k->dtype(), k->device());
            auto grad_v = zeros(v->shape(), v->dtype(), v->device());
            
            return {grad_q, grad_k, grad_v};
        });
    }
    
    return context;
}

} // namespace ops

