/**
 * @file memory_efficient_attention.cpp
 * @brief Memory-efficient attention implementation
 */

#include "memory_efficient_attention.h"
#include "ops.h"
#include "backward_functions.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

namespace {

class MemoryEfficientAttentionBackwardFn : public BackwardFunction {
public:
    MemoryEfficientAttentionBackwardFn(const TensorPtr& q,
                                       const TensorPtr& k,
                                       const TensorPtr& v,
                                       const TensorPtr& causal_mask,
                                       const TensorPtr& pad_mask,
                                       float scale,
                                       bool use_causal_mask)
        : q_(q),
          k_(k),
          v_(v),
          causal_mask_(causal_mask),
          pad_mask_(pad_mask),
          scale_(scale),
          use_causal_mask_(use_causal_mask) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        if (q_->dtype() != kFloat32 || k_->dtype() != kFloat32 || v_->dtype() != kFloat32 ||
            grad_output->dtype() != kFloat32) {
            throw std::runtime_error("memory_efficient_attention backward: only float32 supported");
        }

        const auto& q_shape = q_->shape();
        int64_t batch = q_shape[0];
        int64_t n_head = q_shape[1];
        int64_t seq_len = q_shape[2];
        int64_t head_dim = q_shape[3];

        auto grad_q = zeros(q_->shape(), q_->dtype(), q_->device());
        auto grad_k = zeros(k_->shape(), k_->dtype(), k_->device());
        auto grad_v = zeros(v_->shape(), v_->dtype(), v_->device());

        const float* q_data = q_->data<float>();
        const float* k_data = k_->data<float>();
        const float* v_data = v_->data<float>();
        const float* go_data = grad_output->data<float>();
        float* gq_data = grad_q->data<float>();
        float* gk_data = grad_k->data<float>();
        float* gv_data = grad_v->data<float>();
        const auto& qs = q_->strides();
        const auto& ks = k_->strides();
        const auto& vs = v_->strides();
        const auto& gos = grad_output->strides();
        const auto& gqs = grad_q->strides();
        const auto& gks = grad_k->strides();
        const auto& gvs = grad_v->strides();

        const float* mask_data = nullptr;
        if (causal_mask_) {
            mask_data = causal_mask_->data<float>();
        }
        const float* pad_data = nullptr;
        int pad_ndim = 0;
        if (pad_mask_) {
            pad_ndim = static_cast<int>(pad_mask_->shape().size());
            pad_data = pad_mask_->data<float>();
        }

        static thread_local std::vector<float> scores_buf;
        static thread_local std::vector<float> weights_buf;
        static thread_local std::vector<float> dp_buf;
        if (scores_buf.size() < static_cast<size_t>(seq_len)) scores_buf.resize(seq_len);
        if (weights_buf.size() < static_cast<size_t>(seq_len)) weights_buf.resize(seq_len);
        if (dp_buf.size() < static_cast<size_t>(seq_len)) dp_buf.resize(seq_len);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_head; ++h) {
                for (int64_t i = 0; i < seq_len; ++i) {
                    float* scores = scores_buf.data();
                    float* weights = weights_buf.data();
                    float* dp = dp_buf.data();

                    float max_score = -std::numeric_limits<float>::infinity();

                    for (int64_t j = 0; j < seq_len; ++j) {
                        float score = -1e10f;
                        if (!(use_causal_mask_ && j > i)) {
                            float dot = 0.0f;
                            for (int64_t d = 0; d < head_dim; ++d) {
                                const int64_t q_idx = b * qs[0] + h * qs[1] + i * qs[2] + d * qs[3];
                                const int64_t k_idx = b * ks[0] + h * ks[1] + j * ks[2] + d * ks[3];
                                dot += q_data[q_idx] * k_data[k_idx];
                            }
                            score = dot * scale_;
                            if (mask_data) {
                                score += mask_data[i * seq_len + j];
                            }
                            if (pad_data) {
                                float pad_val = 0.0f;
                                if (pad_ndim == 2) {
                                    pad_val = pad_data[b * seq_len + j];
                                } else {
                                    pad_val = pad_data[b * seq_len + j];
                                }
                                score += pad_val;
                            }
                        }
                        scores[j] = score;
                        if (score > max_score) max_score = score;
                    }

                    double sum_exp = 0.0;
                    for (int64_t j = 0; j < seq_len; ++j) {
                        sum_exp += std::exp(scores[j] - max_score);
                    }
                    float inv_sum = (sum_exp > 0.0) ? (1.0f / static_cast<float>(sum_exp)) : 0.0f;

                    float sum_p_dp = 0.0f;
                    for (int64_t j = 0; j < seq_len; ++j) {
                        float w = std::exp(scores[j] - max_score) * inv_sum;
                        weights[j] = w;

                        float dp_val = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t go_idx = b * gos[0] + h * gos[1] + i * gos[2] + d * gos[3];
                            const int64_t v_idx = b * vs[0] + h * vs[1] + j * vs[2] + d * vs[3];
                            dp_val += go_data[go_idx] * v_data[v_idx];
                        }
                        dp[j] = dp_val;
                        sum_p_dp += w * dp_val;

                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t go_idx = b * gos[0] + h * gos[1] + i * gos[2] + d * gos[3];
                            const int64_t gv_idx = b * gvs[0] + h * gvs[1] + j * gvs[2] + d * gvs[3];
                            gv_data[gv_idx] += w * go_data[go_idx];
                        }
                    }

                    for (int64_t j = 0; j < seq_len; ++j) {
                        float ds = weights[j] * (dp[j] - sum_p_dp);
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t q_idx = b * qs[0] + h * qs[1] + i * qs[2] + d * qs[3];
                            const int64_t k_idx = b * ks[0] + h * ks[1] + j * ks[2] + d * ks[3];
                            const int64_t gq_idx = b * gqs[0] + h * gqs[1] + i * gqs[2] + d * gqs[3];
                            const int64_t gk_idx = b * gks[0] + h * gks[1] + j * gks[2] + d * gks[3];
                            gq_data[gq_idx] += ds * k_data[k_idx] * scale_;
                            gk_data[gk_idx] += ds * q_data[q_idx] * scale_;
                        }
                    }
                }
            }
        }
        return {grad_q, grad_k, grad_v};
    }

private:
    TensorPtr q_;
    TensorPtr k_;
    TensorPtr v_;
    TensorPtr causal_mask_;
    TensorPtr pad_mask_;
    float scale_;
    bool use_causal_mask_;
};

}  // namespace

void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
) {
    // First pass: compute normalization denominator (using max_val for numerical stability)
    double sum_exp = 0.0;
    for (int64_t j = 0; j < seq_len; ++j) {
        sum_exp += std::exp(logits[j] - max_val);
    }
    
    // Second pass: compute normalized weights and accumulate to output
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
    const TensorPtr& pad_mask,
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
    
    // Validate k/v shape matching
    if (k_shape[0] != batch || k_shape[1] != n_head || k_shape[2] != seq_len || k_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: k shape mismatch");
    }
    if (v_shape[0] != batch || v_shape[1] != n_head || v_shape[2] != seq_len || v_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: v shape mismatch");
    }
    
    // Auto-compute scaling factor
    float scale = (config.scale > 0) ? config.scale : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    
    // Prepare causal/base mask (if provided)
    const float* mask_data = nullptr;
    if (causal_mask) {
        if (causal_mask->shape().size() != 2 || 
            causal_mask->shape()[0] != seq_len || 
            causal_mask->shape()[1] != seq_len) {
            throw std::runtime_error("memory_efficient_attention: causal_mask must be [S,S]");
        }
        mask_data = causal_mask->data<float>();
    }

    // Prepare pad mask (if provided)
    const float* pad_data = nullptr;
    int pad_ndim = 0;
    if (pad_mask) {
        pad_ndim = static_cast<int>(pad_mask->shape().size());
        if (pad_ndim != 2 && pad_ndim != 4) {
            throw std::runtime_error("memory_efficient_attention: pad_mask must be [B,S] or [B,1,1,S]");
        }
        if (pad_ndim == 2) {
            if (pad_mask->shape()[0] != batch || pad_mask->shape()[1] != seq_len) {
                throw std::runtime_error("memory_efficient_attention: pad_mask [B,S] shape mismatch");
            }
        } else {
            if (pad_mask->shape()[0] != batch || pad_mask->shape()[3] != seq_len) {
                throw std::runtime_error("memory_efficient_attention: pad_mask [B,1,1,S] shape mismatch");
            }
        }
        pad_data = pad_mask->data<float>();
    }
    
    // Create output tensor
    auto context = zeros({batch, n_head, seq_len, head_dim}, kFloat32, kCPU);
    
    const float* q_data = q->data<float>();
    const float* k_data = k->data<float>();
    const float* v_data = v->data<float>();
    float* ctx_data = context->data<float>();
    const auto& qs = q->strides();
    const auto& ks = k->strides();
    const auto& vs = v->strides();
    const auto& cs = context->strides();
    
    // Main loop: compute per batch, per head, per query row
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            // Process row by row (each query position)
            for (int64_t i = 0; i < seq_len; ++i) {
                // === First pass: compute scores and find max value (numerically stable) ===
                // Reuse same row buffer to avoid heap bloat from repeated allocations
                static thread_local std::vector<float> scores_buf;
                if (scores_buf.size() < static_cast<size_t>(seq_len)) {
                    scores_buf.resize(seq_len);
                }
                float* scores = scores_buf.data();
                float max_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t j = 0; j < seq_len; ++j) {
                    // Compute dot product: q[i] * k[j]
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const int64_t q_idx = b * qs[0] + h * qs[1] + i * qs[2] + d * qs[3];
                        const int64_t k_idx = b * ks[0] + h * ks[1] + j * ks[2] + d * ks[3];
                        dot += q_data[q_idx] * k_data[k_idx];
                    }
                    
                    // Scale
                    float score = dot * scale;
                    
                    // Apply causal mask
                    if (config.use_causal_mask && j > i) {
                        score = -1e10f;  // Mask upper triangle
                    }
                    
                    // Apply additional mask (if provided)
                    if (mask_data) {
                        score += mask_data[i * seq_len + j];
                    }
                    // Apply pad mask (if provided)
                    if (pad_data) {
                        float pad_val = 0.0f;
                        if (pad_ndim == 2) {
                            pad_val = pad_data[b * seq_len + j];
                        } else {
                            // [B,1,1,S] -> index b*S + j
                            pad_val = pad_data[b * seq_len + j];
                        }
                        score += pad_val;
                    }
                    
                    scores[j] = score; // Write to reusable buffer
                    max_score = std::max(max_score, score);
                }
                
                // === Second pass: compute softmax online and accumulate to context ===
                // Initialize output row
                for (int64_t d = 0; d < head_dim; ++d) {
                    const int64_t c_idx = b * cs[0] + h * cs[1] + i * cs[2] + d * cs[3];
                    ctx_data[c_idx] = 0.0f;
                }
                
                // Compute normalization denominator
                double sum_exp = 0.0;
                for (int64_t j = 0; j < seq_len; ++j) {
                    sum_exp += std::exp(scores[j] - max_score);
                }
                
                // Compute weighted sum
                float inv_sum = 1.0f / static_cast<float>(sum_exp);
                for (int64_t j = 0; j < seq_len; ++j) {
                    float weight = std::exp(scores[j] - max_score) * inv_sum;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const int64_t c_idx = b * cs[0] + h * cs[1] + i * cs[2] + d * cs[3];
                        const int64_t v_idx = b * vs[0] + h * vs[1] + j * vs[2] + d * vs[3];
                        ctx_data[c_idx] += weight * v_data[v_idx];
                    }
                }
            }
        }
    }
    
    // Setup gradient propagation (if needed)
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        context->set_requires_grad(true);
        auto backward_fn = std::make_shared<MemoryEfficientAttentionBackwardFn>(
            q, k, v, causal_mask, pad_mask, scale, config.use_causal_mask);

        #ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(context, {q, k, v}, backward_fn);
        #else
        context->set_grad_fn([backward_fn, q, k, v](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (q->requires_grad() && grads.size() > 0 && grads[0]) accumulate_gradient(q, grads[0]);
            if (k->requires_grad() && grads.size() > 1 && grads[1]) accumulate_gradient(k, grads[1]);
            if (v->requires_grad() && grads.size() > 2 && grads[2]) accumulate_gradient(v, grads[2]);
            return grads;
        });
        #endif
    }
    
    return context;
}

} // namespace ops
