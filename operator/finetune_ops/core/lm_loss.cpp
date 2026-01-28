/**
 * @file lm_loss.cpp
 * @brief Language model loss implementation
 */

#include "lm_loss.h"
#include "backward_functions.h"
#include "ops.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

// Cross-entropy backward for language modeling
class LMCrossEntropyBackward : public BackwardFunction {
public:
    LMCrossEntropyBackward(const TensorPtr& logits, const TensorPtr& labels,
                          int ignore_idx, size_t valid_cnt, const std::string& red)
        : logits_(logits), labels_(labels), ignore_index_(ignore_idx),
          valid_count_(valid_cnt), reduction_(red) {}
    
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        // Shapes (HF-style shift: logits[:, :-1] vs labels[:, 1:])
        const auto& shape = logits_->shape();
        int64_t B = shape[0];
        int64_t S = shape[1];
        int64_t V = shape[2];
        int64_t S_eff = (S > 0) ? (S - 1) : 0;
        
        // 创建梯度张量
        auto grad_logits = zeros({B, S, V}, kFloat32, kCPU);
        float* grad_data = grad_logits->data<float>();
        const float* logits_data = logits_->data<float>();
        const int32_t* labels_data = labels_->data<int32_t>();
        
        // Scaling factor (mean normalized by valid tokens)
        float scale_base = 1.0f;
        if (reduction_ == "mean") {
            scale_base = (valid_count_ > 0) ? (1.0f / static_cast<float>(valid_count_)) : 0.0f;
        }
        
        // Per-token gradient (only for the first S-1 logits positions)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S_eff; ++s) {
                int32_t y = labels_data[b * S + (s + 1)];
                if (y == ignore_index_ || y < 0 || y >= V) {
                    // Ignored entry: zero gradient
                    continue;
                }
                
                // Compute softmax at this position (numerically stable)
                const float* logit_row = logits_data + (b * S + s) * V;
                
                // 找最大值
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < V; ++v) {
                    max_val = std::max(max_val, logit_row[v]);
                }
                
                // 计算分母
                float denom = 0.0f;
                for (int64_t v = 0; v < V; ++v) {
                    denom += std::exp(logit_row[v] - max_val);
                }
                
                // Write gradient: grad = (softmax - one_hot(y)) * local_coeff
                float inv_denom = 1.0f / denom;
                // grad_output: mean/sum is scalar; none is per-token weight
                float outer = 1.0f;
                if (grad_output && grad_output->numel() > 0) {
                    if (reduction_ == "none") {
                        outer = grad_output->data<float>()[b * S + s];
                    } else {
                        outer = grad_output->data<float>()[0];
                    }
                }
                float coeff = outer * scale_base;
                
                float* grad_row = grad_data + (b * S + s) * V;
                for (int64_t v = 0; v < V; ++v) {
                    float p = std::exp(logit_row[v] - max_val) * inv_denom;
                    float g = p;
                    if (v == y) {
                        g -= 1.0f;
                    }
                    grad_row[v] = g * coeff;
                }
            }
        }
        
        return {grad_logits};
    }
    
private:
    TensorPtr logits_;
    TensorPtr labels_;
    int ignore_index_;
    size_t valid_count_;
    std::string reduction_;
};

TensorPtr lm_cross_entropy(const TensorPtr& logits,
                           const TensorPtr& labels,
                           int ignore_index,
                           const std::string& reduction) {
    // Supports [B,S,V] × [B,S], normalized by valid tokens (label != ignore_index)
    if (logits->ndim() != 3) {
        throw std::runtime_error("lm_cross_entropy: logits must be [B,S,V]");
    }
    if (labels->ndim() != 2) {
        throw std::runtime_error("lm_cross_entropy: labels must be [B,S]");
    }

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];

    // Forward: numerically stable masked NLL (scalar only; custom Backward provides grads)
    const float* x = logits->data<float>();
    const int32_t* y = labels->data<int32_t>();

    double loss_sum = 0.0;
    int64_t valid_cnt = 0;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s + 1 < S; ++s) {  // shift: logits[:-1] vs labels[1:]
            int32_t cls = y[b * S + (s + 1)];
            if (cls == ignore_index) continue;
            if (cls < 0 || cls >= V) continue;

            const float* row = x + (b * S + s) * V;
            // max for stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) max_val = std::max(max_val, row[v]);
            // logsumexp
            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) denom += std::exp(row[v] - max_val);
            double log_sum_exp = max_val + std::log(denom);
            // NLL
            loss_sum += (log_sum_exp - row[cls]);
            valid_cnt++;
        }
    }

    float out_val = 0.0f;
    if (reduction == "none") {
        // Return [B,S] per-token NLL (invalid positions = 0); computes logits[:-1] × labels[1:]
        auto out = zeros({B, S}, kFloat32, kCPU);
        float* out_data = out->data<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s + 1 < S; ++s) {
                int32_t cls = y[b * S + (s + 1)];
                if (cls == ignore_index || cls < 0 || cls >= V) { out_data[b * S + s] = 0.0f; continue; }
                const float* row = x + (b * S + s) * V;
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < V; ++v) max_val = std::max(max_val, row[v]);
                double denom = 0.0;
                for (int64_t v = 0; v < V; ++v) denom += std::exp(row[v] - max_val);
                double log_sum_exp = max_val + std::log(denom);
                out_data[b * S + s] = static_cast<float>(log_sum_exp - row[cls]);
            }
        }
        // Attach Backward (propagate only to logits; labels do not need grad)
        if (logits->requires_grad()) {
            out->set_requires_grad(true);
            auto backward_fn = std::make_shared<LMCrossEntropyBackward>(logits, labels, ignore_index,
                                                                       static_cast<size_t>(valid_cnt), reduction);
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            autograd::Engine::instance().register_node(out, {logits}, backward_fn);
            #else
            out->set_grad_fn([backward_fn, logits](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grads = backward_fn->apply(grad_output);
                if (logits->requires_grad()) accumulate_gradient(logits, grads[0]);
                return grads;
            });
            #endif
        }
        return out;
    } else if (reduction == "sum" || reduction == "sum_debug") {
        // sum_debug: for alignment/debug; always returns scalar sum and skips valid_count normalization in backward
        out_val = static_cast<float>(loss_sum);
        if (reduction == "sum_debug") {
            valid_cnt = 1;  // backward uses scale_base = 1
        }
    } else { // mean (default)
        out_val = (valid_cnt > 0) ? static_cast<float>(loss_sum / static_cast<double>(valid_cnt)) : 0.0f;
    }

    auto out = full({1}, out_val, kFloat32, kCPU);
    if (logits->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<LMCrossEntropyBackward>(logits, labels, ignore_index,
                                                                   static_cast<size_t>(valid_cnt), reduction);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(out, {logits}, backward_fn);
        #else
        out->set_grad_fn([backward_fn, logits](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (logits->requires_grad()) accumulate_gradient(logits, grads[0]);
            return grads;
        });
        #endif
    }
    return out;
}

}  // namespace ops
