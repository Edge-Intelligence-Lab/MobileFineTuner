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
#include <vector>
#include <utility>

namespace ops {

namespace {

struct ValidShiftedLabel {
    int64_t b;
    int64_t s;
    int32_t cls;
};

std::vector<ValidShiftedLabel> collect_valid_shifted_labels(const TensorPtr& labels,
                                                            int64_t B,
                                                            int64_t S,
                                                            int64_t V,
                                                            int ignore_index) {
    std::vector<ValidShiftedLabel> valid;
    const int32_t* y = labels->data<int32_t>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s + 1 < S; ++s) {
            int32_t cls = y[b * S + (s + 1)];
            if (cls == ignore_index) {
                continue;
            }
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("lm_cross_entropy: label out of range");
            }
            valid.push_back({b, s, cls});
        }
    }
    return valid;
}

float tensor_row_value(const TensorPtr& tensor, int64_t offset) {
    if (tensor->dtype() == kFloat32) {
        return tensor->data<float>()[offset];
    }
    if (tensor->dtype() == kFloat16) {
        return fp16_bits_to_float32(tensor->data<uint16_t>()[offset]);
    }
    if (tensor->dtype() == kBFloat16) {
        return bf16_bits_to_float32(tensor->data<uint16_t>()[offset]);
    }
    throw std::runtime_error("lm_loss: unsupported floating weight dtype");
}

float dot_hidden_weight(const float* hidden_row, const TensorPtr& weight, int64_t row, int64_t H) {
    float sum = 0.0f;
    const int64_t base = row * H;
    for (int64_t h = 0; h < H; ++h) {
        sum += hidden_row[h] * tensor_row_value(weight, base + h);
    }
    return sum;
}

} // namespace

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
                if (y == ignore_index_) {
                    // Ignored entry: zero gradient
                    continue;
                }
                if (y < 0 || y >= V) {
                    throw std::runtime_error("lm_cross_entropy backward: label out of range");
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

class SelectedTokenLMCrossEntropyBackward : public BackwardFunction {
public:
    SelectedTokenLMCrossEntropyBackward(const TensorPtr& hidden,
                                        const TensorPtr& lm_head_weight,
                                        const TensorPtr& labels,
                                        std::vector<ValidShiftedLabel> valid,
                                        const std::string& red)
        : hidden_(hidden),
          lm_head_weight_(lm_head_weight),
          labels_(labels),
          valid_(std::move(valid)),
          reduction_(red) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        (void)labels_;

        const auto& hshape = hidden_->shape();
        const auto& wshape = lm_head_weight_->shape();
        const int64_t B = hshape[0];
        const int64_t S = hshape[1];
        const int64_t H = hshape[2];
        const int64_t V = wshape[0];

        TensorPtr grad_hidden = hidden_->requires_grad()
            ? zeros({B, S, H}, kFloat32, kCPU)
            : nullptr;
        TensorPtr grad_weight = lm_head_weight_->requires_grad()
            ? zeros({V, H}, kFloat32, kCPU)
            : nullptr;

        if ((!grad_hidden && !grad_weight) || valid_.empty()) {
            return {grad_hidden, grad_weight, nullptr};
        }

        float outer = 1.0f;
        if (grad_output && grad_output->numel() > 0) {
            outer = grad_output->data<float>()[0];
        }

        float scale_base = 1.0f;
        if (reduction_ == "mean") {
            scale_base = 1.0f / static_cast<float>(valid_.size());
        }
        const float coeff = outer * scale_base;
        if (coeff == 0.0f) {
            return {grad_hidden, grad_weight, nullptr};
        }

        const float* hidden_data = hidden_->data<float>();
        float* grad_hidden_data = grad_hidden ? grad_hidden->data<float>() : nullptr;
        float* grad_weight_data = grad_weight ? grad_weight->data<float>() : nullptr;

        std::vector<float> logits(static_cast<size_t>(V));
        for (const auto& item : valid_) {
            const float* hrow = hidden_data + (item.b * S + item.s) * H;

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                const float z = dot_hidden_weight(hrow, lm_head_weight_, v, H);
                logits[static_cast<size_t>(v)] = z;
                max_val = std::max(max_val, z);
            }

            float denom = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(logits[static_cast<size_t>(v)] - max_val);
            }
            const float inv_denom = 1.0f / denom;

            float* ghrow = grad_hidden_data ? grad_hidden_data + (item.b * S + item.s) * H : nullptr;
            for (int64_t v = 0; v < V; ++v) {
                float p = std::exp(logits[static_cast<size_t>(v)] - max_val) * inv_denom;
                if (v == item.cls) {
                    p -= 1.0f;
                }
                const float g = p * coeff;

                if (ghrow) {
                    const int64_t wbase = v * H;
                    for (int64_t h = 0; h < H; ++h) {
                        ghrow[h] += g * tensor_row_value(lm_head_weight_, wbase + h);
                    }
                }
                if (grad_weight_data) {
                    float* gwrow = grad_weight_data + v * H;
                    for (int64_t h = 0; h < H; ++h) {
                        gwrow[h] += g * hrow[h];
                    }
                }
            }
        }

        return {grad_hidden, grad_weight, nullptr};
    }

private:
    TensorPtr hidden_;
    TensorPtr lm_head_weight_;
    TensorPtr labels_;
    std::vector<ValidShiftedLabel> valid_;
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
    if (labels->shape()[0] != B || labels->shape()[1] != S) {
        throw std::runtime_error("lm_cross_entropy: labels shape must match logits [B,S]");
    }

    // Forward: numerically stable masked NLL (scalar only; custom Backward provides grads)
    const float* x = logits->data<float>();
    const int32_t* y = labels->data<int32_t>();

    double loss_sum = 0.0;
    int64_t valid_cnt = 0;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s + 1 < S; ++s) {  // shift: logits[:-1] vs labels[1:]
            int32_t cls = y[b * S + (s + 1)];
            if (cls == ignore_index) continue;
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("lm_cross_entropy: label out of range");
            }

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
                if (cls == ignore_index) { out_data[b * S + s] = 0.0f; continue; }
                if (cls < 0 || cls >= V) {
                    throw std::runtime_error("lm_cross_entropy: label out of range");
                }
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
        out_val = (valid_cnt > 0)
            ? static_cast<float>(loss_sum / static_cast<double>(valid_cnt))
            : std::numeric_limits<float>::quiet_NaN();
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

TensorPtr selected_token_lm_cross_entropy(const TensorPtr& hidden,
                                         const TensorPtr& lm_head_weight,
                                         const TensorPtr& labels,
                                         int ignore_index,
                                         const std::string& reduction) {
    if (!hidden || !lm_head_weight || !labels) {
        throw std::runtime_error("selected_token_lm_cross_entropy: inputs must not be null");
    }
    if (hidden->ndim() != 3) {
        throw std::runtime_error("selected_token_lm_cross_entropy: hidden must be [B,S,H]");
    }
    if (lm_head_weight->ndim() != 2) {
        throw std::runtime_error("selected_token_lm_cross_entropy: lm_head_weight must be [V,H]");
    }
    if (labels->ndim() != 2) {
        throw std::runtime_error("selected_token_lm_cross_entropy: labels must be [B,S]");
    }
    if (reduction != "mean" && reduction != "sum") {
        throw std::runtime_error("selected_token_lm_cross_entropy: only mean and sum reductions are supported");
    }

    const auto& hshape = hidden->shape();
    const auto& wshape = lm_head_weight->shape();
    const int64_t B = hshape[0];
    const int64_t S = hshape[1];
    const int64_t H = hshape[2];
    const int64_t V = wshape[0];

    if (wshape[1] != H) {
        throw std::runtime_error("selected_token_lm_cross_entropy: hidden H and weight H mismatch");
    }
    if (labels->shape()[0] != B || labels->shape()[1] != S) {
        throw std::runtime_error("selected_token_lm_cross_entropy: labels shape must match hidden [B,S]");
    }

    auto valid = collect_valid_shifted_labels(labels, B, S, V, ignore_index);
    double loss_sum = 0.0;
    const float* hidden_data = hidden->data<float>();

    for (const auto& item : valid) {
        const float* hrow = hidden_data + (item.b * S + item.s) * H;

        float max_val = -std::numeric_limits<float>::infinity();
        float target_logit = 0.0f;
        for (int64_t v = 0; v < V; ++v) {
            const float z = dot_hidden_weight(hrow, lm_head_weight, v, H);
            if (v == item.cls) {
                target_logit = z;
            }
            max_val = std::max(max_val, z);
        }

        double denom = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            const float z = dot_hidden_weight(hrow, lm_head_weight, v, H);
            denom += std::exp(z - max_val);
        }
        const double log_sum_exp = max_val + std::log(denom);
        loss_sum += (log_sum_exp - target_logit);
    }

    float out_val = 0.0f;
    if (reduction == "sum") {
        out_val = static_cast<float>(loss_sum);
    } else {
        out_val = valid.empty()
            ? std::numeric_limits<float>::quiet_NaN()
            : static_cast<float>(loss_sum / static_cast<double>(valid.size()));
    }

    auto out = full({1}, out_val, kFloat32, kCPU);
    if (hidden->requires_grad() || lm_head_weight->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<SelectedTokenLMCrossEntropyBackward>(
            hidden, lm_head_weight, labels, valid, reduction);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(out, {hidden, lm_head_weight, labels}, backward_fn);
        #else
        out->set_grad_fn([backward_fn, hidden, lm_head_weight](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (hidden->requires_grad() && grads[0]) accumulate_gradient(hidden, grads[0]);
            if (lm_head_weight->requires_grad() && grads[1]) accumulate_gradient(lm_head_weight, grads[1]);
            return grads;
        });
        #endif
    }

    return out;
}

TensorPtr streaming_lm_cross_entropy(const TensorPtr& hidden,
                                    const TensorPtr& lm_head_weight,
                                    const TensorPtr& labels,
                                    int ignore_index,
                                    const std::string& reduction) {
    return selected_token_lm_cross_entropy(
        hidden, lm_head_weight, labels, ignore_index, reduction);
}

}  // namespace ops
