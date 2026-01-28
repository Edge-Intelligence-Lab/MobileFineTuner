/**
 * @file chunked_softmax_ce.cpp
 * @brief Accurate chunked Softmax + CrossEntropy implementsation
 */

#include "chunked_softmax_ce.h"
#include "tensor.h"
#include "backward_functions.h"
#include "autograd_engine.h"
#include <cstring>
#include <iostream>

namespace ops {
namespace chunked_ce {

// ChunkedCEbackwardfunctionclass
class ChunkedCEBackward : public BackwardFunction {
private:
    TensorPtr X_, W_, targets_;
    int64_t chunk_size_;
    bool W_is_transposed_;
    
public:
    ChunkedCEBackward(const TensorPtr& X, const TensorPtr& W, 
                     const TensorPtr& targets, int64_t chunk_size, bool W_is_transposed)
        : X_(X), W_(W), targets_(targets), chunk_size_(chunk_size), W_is_transposed_(W_is_transposed) {}
    
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        float grad_val = grad_output->data<float>()[0];
        auto [grad_X, grad_W] = chunked_cross_entropy_backward(
            X_, W_, targets_, grad_val, chunk_size_, W_is_transposed_);
        return {grad_X, grad_W};
    }
};

TensorPtr chunked_cross_entropy_forward(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    int64_t chunk_size,
    bool W_is_transposed
) {
    auto X_shape = X->shape();
    auto W_shape = W->shape();
    auto tgt_shape = targets->shape();
    
    // Validate shapes
    if (X_shape.size() != 3) {
        throw std::runtime_error("chunked_ce: X requires [B, L, D]");
    }
    if (tgt_shape.size() != 2) {
        throw std::runtime_error("chunked_ce: targets requires [B, L]");
    }
    
    int64_t B = X_shape[0];
    int64_t L = X_shape[1];
    int64_t D = X_shape[2];
    int64_t V = W_is_transposed ? W_shape[1] : W_shape[0];
    
    if (B != tgt_shape[0] || L != tgt_shape[1]) {
        throw std::runtime_error("chunked_ce: X batch/sequence dimensions mismatch with targets");
    }
    
    const float* X_data = X->data<float>();
    const float* W_data = W->data<float>();
    const int32_t* tgt_data = targets->data<int32_t>();
    
    // Accumulate loss
    double total_loss = 0.0;
    int64_t total_count = B * L;
    
    // Temporary buffer: logits_chunk
    std::vector<float> logits_chunk(chunk_size);
    
    // For each sample and position
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t l = 0; l < L; ++l) {
            int64_t idx = b * L + l;
            int32_t target_class = tgt_data[idx];
            
            // Skip invalid classes (e.g., padding)
            if (target_class < 0 || target_class >= V) {
                continue;
            }
            
            const float* x_vec = X_data + idx * D;  // [D]
            
            // Streaming LogSumExp state
            StreamingLogSumExpState lse_state;
            float target_logit = -std::numeric_limits<float>::infinity();
            
            // Iterate through vocabulary in chunks
            int64_t num_chunks = (V + chunk_size - 1) / chunk_size;
            for (int64_t c = 0; c < num_chunks; ++c) {
                int64_t chunk_start = c * chunk_size;
                int64_t chunk_end = std::min(chunk_start + chunk_size, V);
                int64_t current_chunk_size = chunk_end - chunk_start;
                
                // Compute logits_chunk = x @ W_chunk^T
                for (int64_t i = 0; i < current_chunk_size; ++i) {
                    int64_t vocab_idx = chunk_start + i;
                    float dot = 0.0f;
                    
                    if (W_is_transposed) {
                        // W shape [D, V]
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * W_data[d * V + vocab_idx];
                        }
                    } else {
                        // W shape [V, D]
                        const float* w_row = W_data + vocab_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * w_row[d];
                        }
                    }
                    
                    logits_chunk[i] = dot;
                    
                    // If this is the target class, record its logit
                    if (vocab_idx == target_class) {
                        target_logit = dot;
                    }
                }
                
                // Update streaming logsumexp
                lse_state.update(logits_chunk.data(), current_chunk_size);
            }
            
            // Calculate loss for this position
            float log_sum_exp = lse_state.get_log_sum_exp();
            if (target_logit == -std::numeric_limits<float>::infinity()) {
                // 目標類不在任何chunk中：回退為直接使用 log_sum_exp（等價於忽略該位置）
                // 或者你可以選擇丟棄/報警告；這裡選擇加和但不產生NaN
                // std::cerr << "[chunked_ce] target class not seen in chunks at ("<<b<<","<<l<<")"<<std::endl;
                total_loss += log_sum_exp; // 相當於 target_logit=0 情況下的上界
            } else {
                float loss_val = -(target_logit - log_sum_exp);
                total_loss += loss_val;
            }
        }
    }
    
    // Return average loss
    float mean_loss = static_cast<float>(total_loss / total_count);
    auto loss_tensor = zeros({1});
    loss_tensor->data<float>()[0] = mean_loss;
    
    return loss_tensor;
}

std::pair<TensorPtr, TensorPtr> chunked_cross_entropy_backward(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    float grad_output,
    int64_t chunk_size,
    bool W_is_transposed
) {
    auto X_shape = X->shape();
    auto W_shape = W->shape();
    
    int64_t B = X_shape[0];
    int64_t L = X_shape[1];
    int64_t D = X_shape[2];
    int64_t V = W_is_transposed ? W_shape[1] : W_shape[0];
    
    const float* X_data = X->data<float>();
    const float* W_data = W->data<float>();
    const int32_t* tgt_data = targets->data<int32_t>();
    
    // Initialize gradients
    auto grad_X = zeros(X_shape);
    auto grad_W = zeros(W_shape);
    
    float* grad_X_data = grad_X->data<float>();
    float* grad_W_data = grad_W->data<float>();
    
    // Normalization factor
    int64_t total_count = B * L;
    float scale = grad_output / static_cast<float>(total_count);
    
    // Temporary buffer
    std::vector<float> logits_chunk(chunk_size);
    std::vector<float> softmax_chunk(chunk_size);
    
    // For each sample and position
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t l = 0; l < L; ++l) {
            int64_t idx = b * L + l;
            int32_t target_class = tgt_data[idx];
            
            if (target_class < 0 || target_class >= V) {
                continue;
            }
            
            const float* x_vec = X_data + idx * D;
            float* grad_x_vec = grad_X_data + idx * D;
            
            // First pass: Calculate logsumexp
            StreamingLogSumExpState lse_state;
            int64_t num_chunks = (V + chunk_size - 1) / chunk_size;
            
            for (int64_t c = 0; c < num_chunks; ++c) {
                int64_t chunk_start = c * chunk_size;
                int64_t chunk_end = std::min(chunk_start + chunk_size, V);
                int64_t current_chunk_size = chunk_end - chunk_start;
                
                for (int64_t i = 0; i < current_chunk_size; ++i) {
                    int64_t vocab_idx = chunk_start + i;
                    float dot = 0.0f;
                    
                    if (W_is_transposed) {
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * W_data[d * V + vocab_idx];
                        }
                    } else {
                        const float* w_row = W_data + vocab_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * w_row[d];
                        }
                    }
                    logits_chunk[i] = dot;
                }
                
                lse_state.update(logits_chunk.data(), current_chunk_size);
            }
            
            float log_sum_exp = lse_state.get_log_sum_exp();
            
            // Second pass: Calculate gradients (chunk by chunk)
            for (int64_t c = 0; c < num_chunks; ++c) {
                int64_t chunk_start = c * chunk_size;
                int64_t chunk_end = std::min(chunk_start + chunk_size, V);
                int64_t current_chunk_size = chunk_end - chunk_start;
                
                // Recompute logits_chunk
                for (int64_t i = 0; i < current_chunk_size; ++i) {
                    int64_t vocab_idx = chunk_start + i;
                    float dot = 0.0f;
                    
                    if (W_is_transposed) {
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * W_data[d * V + vocab_idx];
                        }
                    } else {
                        const float* w_row = W_data + vocab_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            dot += x_vec[d] * w_row[d];
                        }
                    }
                    logits_chunk[i] = dot;
                }
                
                // Calculate softmax_chunk
                for (int64_t i = 0; i < current_chunk_size; ++i) {
                    softmax_chunk[i] = std::exp(logits_chunk[i] - log_sum_exp);
                }
                
                                // [Translated]
                for (int64_t i = 0; i < current_chunk_size; ++i) {
                    int64_t vocab_idx = chunk_start + i;
                    
                    // (p - y)
                    float grad_logit = softmax_chunk[i];
                    if (vocab_idx == target_class) {
                        grad_logit -= 1.0f;
                    }
                    grad_logit *= scale;
                    
                    // grad_W accumulate：W.grad[vocab_idx] += grad_logit * x
                    if (W_is_transposed) {
                        // W shape [D, V]
                        for (int64_t d = 0; d < D; ++d) {
                            grad_W_data[d * V + vocab_idx] += grad_logit * x_vec[d];
                        }
                    } else {
                        // W shape [V, D]
                        float* grad_w_row = grad_W_data + vocab_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            grad_w_row[d] += grad_logit * x_vec[d];
                        }
                    }
                    
                    // grad_X accumulate：X.grad += grad_logit * W[vocab_idx]
                    if (W_is_transposed) {
                        for (int64_t d = 0; d < D; ++d) {
                            grad_x_vec[d] += grad_logit * W_data[d * V + vocab_idx];
                        }
                    } else {
                        const float* w_row = W_data + vocab_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            grad_x_vec[d] += grad_logit * w_row[d];
                        }
                    }
                }
            }
        }
    }
    
    return {grad_X, grad_W};
}

TensorPtr chunked_cross_entropy_loss(
    const TensorPtr& X,
    const TensorPtr& W,
    const TensorPtr& targets,
    int64_t chunk_size
) {
    // Determine if W is transposed (by shape)
    auto W_shape = W->shape();
    auto X_shape = X->shape();
    int64_t D = X_shape[2];
    
    bool W_is_transposed = (W_shape[0] == D);  // If first dimension is D, shape is [D, V]
    
    // Forward computation
    auto loss = chunked_cross_entropy_forward(X, W, targets, chunk_size, W_is_transposed);
    
    // Register backward function
    if (X->requires_grad() || W->requires_grad()) {
        loss->set_requires_grad(true);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // New engine: Create and register BackwardFunction
        auto backward_fn = std::make_shared<ChunkedCEBackward>(X, W, targets, chunk_size, W_is_transposed);
        autograd::Engine::instance().register_node(loss, {X, W}, backward_fn);
        #else
        // Old engine: Use set_grad_fn to set backward function
        loss->set_grad_fn([X, W, targets, chunk_size, W_is_transposed](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            float grad_val = grad_output->data<float>()[0];
            
            auto [grad_X, grad_W] = chunked_cross_entropy_backward(
                X, W, targets, grad_val, chunk_size, W_is_transposed);
            
            return {grad_X, grad_W};
        });
        #endif
    }
    
    return loss;
}

} // namespace chunked_ce
} // namespace ops

