/**
 * @file chunked_softmax_ce.h
 * @brief accuratechunked Softmax + CrossEntropy（containbackwardpropagate）
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * 
 * [Documentation in English - see separate docs]
 */

#pragma once

#include "tensor.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {
namespace chunked_ce {

/**
 * @brief Streaming LogSumExp state
 */
struct StreamingLogSumExpState {
    float max_logit = -std::numeric_limits<float>::infinity();
    double sum_exp = 0.0; // 使用 double 累積，降低精度損失
    
    void update(const float* logits, int64_t size) {
        float local_max = max_logit;
        for (int64_t i = 0; i < size; ++i) {
            local_max = std::max(local_max, logits[i]);
        }
        
        if (local_max > max_logit) {
            sum_exp *= std::exp(static_cast<double>(max_logit - local_max));
            max_logit = local_max;
        }
        
        for (int64_t i = 0; i < size; ++i) {
            sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
        }
    }
    
    float get_log_sum_exp() const {
        return static_cast<float>(max_logit + std::log(sum_exp));
    }
};

/**
 * [Documentation available in English]
 * 
 * @param X inputfeature [B, L, D]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation available in English]
 * @return loss scalar
 * 
 * algorithm：
 * 1. foreach (b, l)，chunkedCompute logits_chunk = X[b,l] @ W_chunk
 * [Documentation available in English]
 * [Documentation available in English]
 * 4. final loss = -mean(target_logit - logsumexp)
 */
TensorPtr chunked_cross_entropy_forward(
    const TensorPtr& X,           // [B, L, D]
    const TensorPtr& W,           // [V, D] or [D, V]
    const TensorPtr& targets,     // [B, L] int32
    int64_t chunk_size = 2048,
    bool W_is_transposed = false
);

/**
 * [Documentation available in English]
 * 
 * @param X inputfeature [B, L, D]
 * @param W lm_head weight [V, D] or [D, V]
 * [Documentation available in English]
 * [Documentation available in English]
 * @param chunk_size chunkedsize
 * [Documentation available in English]
 * @return (grad_X, grad_W) gradienttensorfor
 * 
 * algorithm：
 * [Documentation available in English]
 * 2. Compute (p_chunk - y_one_hot_chunk) * grad_output
 * 3. accumulate W.grad += X^T @ (p - y)
 * 4. accumulate X.grad += (p - y) @ W^T
 */
std::pair<TensorPtr, TensorPtr> chunked_cross_entropy_backward(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    float grad_output,
    int64_t chunk_size = 2048,
    bool W_is_transposed = false
);

/**
 * [Documentation available in English]
 * 
 * @param X inputfeature [B, L, D]
 * @param W lm_head weight（support [V,D] or [D,V]）
 * [Documentation available in English]
 * @param chunk_size chunkedsize
 * @return loss scalartensor（support backward）
 */
TensorPtr chunked_cross_entropy_loss(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    int64_t chunk_size = 2048
);

} // namespace chunked_ce
} // namespace ops

