/**
 * @file lora_linear.cpp
 * @brief LoRA-augmented linear layer (manual LoRA-branch backward; A/B trainable)
 */

#include "lora_linear.h"
#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include "../core/backward_functions.h"
#include <stdexcept>
#include <cstdlib>
#include <iostream>

namespace ops {

namespace {
// Hand-written LoRA branch backward: only handles x/A/B
struct LoRADeltaBackward : public BackwardFunction {
    TensorPtr x_;
    TensorPtr A_eff_;  // [in, r]
    TensorPtr B_eff_;  // [r, out]
    TensorPtr A_orig_;
    TensorPtr B_orig_;
    float scale_;
    bool a_transposed_;  // Whether original A was [r,in]
    bool b_transposed_;  // Whether original B was [out,r]
    int64_t batch_, seq_, in_dim_, out_dim_, rank_;
    std::string debug_name_;

    LoRADeltaBackward(const TensorPtr& x,
                      const TensorPtr& A_eff,
                      const TensorPtr& B_eff,
                      const TensorPtr& A_orig,
                      const TensorPtr& B_orig,
                      float scale,
                      bool a_t,
                      bool b_t,
                      const std::string& dbg_name)
        : x_(x), A_eff_(A_eff), B_eff_(B_eff), A_orig_(A_orig), B_orig_(B_orig),
          scale_(scale), a_transposed_(a_t), b_transposed_(b_t), debug_name_(dbg_name) {
        const auto& xs = x_->shape();
        batch_ = xs[0]; seq_ = xs[1]; in_dim_ = xs[2];
        rank_ = A_eff_->shape()[1];
        out_dim_ = B_eff_->shape()[1];
    }

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        const bool debug = std::getenv("LORA_LINEAR_TEST_DEBUG") != nullptr;
        const auto& gos = grad_output->shape();
        if (gos.size() != 3) {
            throw std::runtime_error("LoRADeltaBackward: grad_output rank must be 3");
        }
        int64_t N = batch_ * seq_;

        const float* x_data = x_->data<float>();
        const float* A_data = A_eff_->data<float>(); // [in,r]
        const float* B_data = B_eff_->data<float>(); // [r,out]
        const float* go_data = grad_output->data<float>();

        // H = X @ A  [N, r]
        std::vector<float> H(static_cast<size_t>(N * rank_), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t j = 0; j < rank_; ++j) {
                double sum = 0.0;
                for (int64_t d = 0; d < in_dim_; ++d) {
                    sum += static_cast<double>(x_data[n * in_dim_ + d]) *
                           static_cast<double>(A_data[d * rank_ + j]);
                }
                H[static_cast<size_t>(n * rank_ + j)] = static_cast<float>(sum);
            }
        }

        // grad_B = scale * (G^T @ H)  => [out, r]
        auto grad_B_out_r = zeros({out_dim_, rank_}, DType::kFloat32, grad_output->device());
        float* gB = grad_B_out_r->data<float>();
        for (int64_t o = 0; o < out_dim_; ++o) {
            for (int64_t j = 0; j < rank_; ++j) {
                double sum = 0.0;
                for (int64_t n = 0; n < N; ++n) {
                    sum += static_cast<double>(go_data[n * out_dim_ + o]) *
                           static_cast<double>(H[n * rank_ + j]);
                }
                gB[o * rank_ + j] = static_cast<float>(sum * static_cast<double>(scale_));
            }
        }
        auto grad_B_eff = transpose(grad_B_out_r, 0, 1); // [r,out]
        TensorPtr grad_B;
        if (b_transposed_) {
            grad_B = transpose(grad_B_eff, 0, 1); // Return to [out,r]
        } else {
            grad_B = grad_B_eff;
        }

        // grad_A = scale * X^T @ (G @ B^T) => [in, r]
        auto grad_A = zeros({in_dim_, rank_}, DType::kFloat32, grad_output->device());
        float* gA = grad_A->data<float>();
        for (int64_t d = 0; d < in_dim_; ++d) {
            for (int64_t j = 0; j < rank_; ++j) {
                double sum = 0.0;
                for (int64_t n = 0; n < N; ++n) {
                    double tmp = 0.0;
                    for (int64_t o = 0; o < out_dim_; ++o) {
                        tmp += static_cast<double>(go_data[n * out_dim_ + o]) *
                               static_cast<double>(B_data[j * out_dim_ + o]);
                    }
                    sum += static_cast<double>(x_data[n * in_dim_ + d]) * tmp;
                }
                gA[d * rank_ + j] = static_cast<float>(sum * static_cast<double>(scale_));
            }
        }
        TensorPtr grad_A_final = a_transposed_ ? transpose(grad_A, 0, 1) : grad_A;

        // grad_input = scale * (G @ B^T @ A^T)  => reshape [B,S,in]
        TensorPtr grad_input = nullptr;
        if (x_->requires_grad()) {
            auto grad_in = zeros({N, in_dim_}, DType::kFloat32, grad_output->device());
            float* gI = grad_in->data<float>();
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < in_dim_; ++d) {
                    double sum = 0.0;
                    for (int64_t j = 0; j < rank_; ++j) {
                        double tmp = 0.0;
                        for (int64_t o = 0; o < out_dim_; ++o) {
                            tmp += static_cast<double>(go_data[n * out_dim_ + o]) *
                                   static_cast<double>(B_data[j * out_dim_ + o]);
                        }
                        sum += tmp * static_cast<double>(A_data[d * rank_ + j]);
                    }
                    gI[n * in_dim_ + d] = static_cast<float>(sum * static_cast<double>(scale_));
                }
            }
            grad_input = reshape(grad_in, {batch_, seq_, in_dim_});
        }

        if (debug) {
            double max_gB = 0.0, max_gA = 0.0;
            const float* gB_data = grad_B->data<float>();
            const float* gA_data = grad_A_final->data<float>();
            for (int64_t i = 0; i < grad_B->numel(); ++i) {
                double v = std::abs(static_cast<double>(gB_data[i]));
                if (v > max_gB) max_gB = v;
            }
            for (int64_t i = 0; i < grad_A_final->numel(); ++i) {
                double v = std::abs(static_cast<double>(gA_data[i]));
                if (v > max_gA) max_gA = v;
            }
            std::cerr << "[LoRADeltaBackward] grad_A_max=" << max_gA
                      << " grad_B_max=" << max_gB;
            if (!debug_name_.empty()) std::cerr << " name=" << debug_name_;
            if (debug_name_.find("q_proj") != std::string::npos ||
                debug_name_.find("k_proj") != std::string::npos ||
                debug_name_.find("v_proj") != std::string::npos) {
                double max_go = 0.0;
                const float* go_data = grad_output->data<float>();
                for (int64_t i = 0; i < grad_output->numel(); ++i) {
                    double v = std::abs(static_cast<double>(go_data[i]));
                    if (v > max_go) max_go = v;
                }
                std::cerr << " go_max=" << max_go;
            }
            std::cerr << std::endl;
        }

        return {grad_input, grad_A_final, grad_B};
    }
};
}  // namespace

void LoRALinear::attach_lora(const TensorPtr& A, const TensorPtr& B,
                             float scale, int col0, int cols) {
    if (!A || !B) {
        throw std::runtime_error("LoRALinear::attach_lora: A and B must not be null");
    }
    
    // Shape checks
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

std::vector<std::pair<std::string, TensorPtr>> LoRALinear::debug_params() const {
    std::vector<std::pair<std::string, TensorPtr>> result;
    for (size_t i = 0; i < slices_.size(); ++i) {
        std::string prefix = debug_name_.empty() ? ("lora_" + std::to_string(i)) : debug_name_;
        result.emplace_back(prefix + "_lora_A_default_weight", slices_[i].A);
        result.emplace_back(prefix + "_lora_B_default_weight", slices_[i].B);
    }
    return result;
}

TensorPtr LoRALinear::forward(const TensorPtr& x) const {
    const bool debug = std::getenv("LORA_LINEAR_TEST_DEBUG") != nullptr;
    auto log_shape = [&](const std::string& tag, const TensorPtr& a, const TensorPtr& b) {
        if (!debug) return;
        std::cerr << "[LoRALinear] " << tag
                  << " A_shape=";
        if (a) {
            const auto& s = a->shape();
            std::cerr << "[";
            for (size_t i = 0; i < s.size(); ++i) {
                std::cerr << s[i] << (i + 1 == s.size() ? "" : ",");
            }
            std::cerr << "]";
        } else {
            std::cerr << "null";
        }
        std::cerr << " B_shape=";
        if (b) {
            const auto& s = b->shape();
            std::cerr << "[";
            for (size_t i = 0; i < s.size(); ++i) {
                std::cerr << s[i] << (i + 1 == s.size() ? "" : ",");
            }
            std::cerr << "]";
        } else {
            std::cerr << "null";
        }
        std::cerr << " A_ptr=" << a.get() << " B_ptr=" << b.get() << std::endl;
    };

    // Base branch: x @ W + b (base frozen, no gradients)
    log_shape("base matmul", x, W_);
    TensorPtr y;
    try {
        y = matmul(x, W_);
    } catch (const std::exception& e) {
        if (debug) {
            std::cerr << "[LoRALinear] base matmul threw: " << e.what() << std::endl;
        }
        throw;
    }
    if (b_) {
        y = add(y, b_);
    }
    
    // LoRA delta: Σ scale * (x @ A @ B)
    if (!slices_.empty() && !merged_) {
        const auto& wshape = W_->shape();
        int64_t w0 = wshape[0];
        int64_t w1 = wshape[1];
        int64_t in_dim_x = x->shape()[2];
        // Handle W_ shaped as either [out,in] or [in,out]
        int64_t out_dim = (w1 == in_dim_x) ? w0 : (w0 == in_dim_x ? w1 : w0);
        for (const auto& slice : slices_) {
            if (!slice.A->requires_grad()) {
                const_cast<Tensor*>(slice.A.get())->set_requires_grad(true);
            }
            if (!slice.B->requires_grad()) {
                const_cast<Tensor*>(slice.B.get())->set_requires_grad(true);
            }
            
            // Normalize A/B shapes: A expects [in, r], B expects [r, out]
            TensorPtr A_eff = slice.A;
            bool a_transposed = false;
            if (A_eff->shape()[0] != x->shape()[2] && A_eff->shape()[1] == x->shape()[2]) {
                A_eff = transpose(A_eff, 0, 1);
                a_transposed = true;
            }
            int64_t r = A_eff->shape()[1];
            TensorPtr B_eff = slice.B;
            bool b_transposed = false;
            if (B_eff->shape()[0] != r && B_eff->shape()[1] == r) {
                B_eff = transpose(B_eff, 0, 1);
                b_transposed = true;
            }
            if (B_eff->shape()[0] != r) {
                if (debug) {
                    std::cerr << "[LoRALinear] B shape mismatch: "
                              << "B=";
                    const auto& sb = B_eff->shape();
                    std::cerr << "[";
                    for (size_t i = 0; i < sb.size(); ++i) {
                        std::cerr << sb[i] << (i + 1 == sb.size() ? "" : ",");
                    }
                    std::cerr << "], r=" << r
                              << " a_transposed=" << a_transposed
                              << " b_transposed=" << b_transposed
                              << std::endl;
                }
                throw std::runtime_error("LoRA B shape mismatch");
            }

            auto x_shape = x->shape();
            int64_t batch = x_shape[0];
            int64_t seq = x_shape[1];
            int64_t in_dim = x_shape[2];

            // delta = (x @ A_eff) @ B_eff
            auto X2d = reshape(x, {batch * seq, in_dim});        // [N, in]
            log_shape("xA matmul", X2d, A_eff);
            auto xA2d = matmul(X2d, A_eff);                      // [N, r]
            log_shape("delta matmul", xA2d, B_eff);
            auto delta2d = matmul(xA2d, B_eff);                  // [N, out_slice]
            delta2d = mul(delta2d, slice.scale);
            auto delta = reshape(delta2d, {batch, seq, B_eff->shape()[1]});
            
            if (slice.cols == out_dim && slice.col0 == 0) {
                y = add(y, delta);
            } else {
                // 部分列更新
                float* y_data = y->data<float>();
                const float* delta_data = delta->data<float>();
                auto y_shape = y->shape();
                int64_t Bn = y_shape[0];
                int64_t S = y_shape[1];
                for (int64_t b = 0; b < Bn; ++b) {
                    for (int64_t s = 0; s < S; ++s) {
                        for (int64_t c = 0; c < slice.cols; ++c) {
                            int64_t y_idx = b * S * out_dim + s * out_dim + (slice.col0 + c);
                            int64_t delta_idx = b * S * slice.cols + s * slice.cols + c;
                            y_data[y_idx] += delta_data[delta_idx];
                        }
                    }
                }
            }

            // Rely on standard autograd path: delta = ((x @ A_eff) @ B_eff) * scale
            // A_eff/B_eff are derived via transpose and have registered backward to propagate to original A/B
        }
    }
    
    return y;
}

void LoRALinear::merge_to_base() {
    if (merged_ || slices_.empty()) return;
    
    // Compute ΔW for all LoRA slices and add to base
    for (const auto& slice : slices_) {
        // ΔW_slice = A @ B * scale  -> [in, out_slice]
        auto delta_W = matmul(slice.A, slice.B);
        delta_W = mul(delta_W, slice.scale);
        
        // Add in-place to W submatrix W[:, col0:col0+cols]
        // Simplified: add to full W (assumes slice.col0=0 and cols cover full width)
        // TODO: full implementation should use submatrix ops
        float* W_data = W_->data<float>();
        const float* delta_data = delta_W->data<float>();
        
        int64_t in_dim = W_->shape()[0];
        int64_t out_dim = W_->shape()[1];
        
        // Validate: for non-split cases delta should match W shape
        if (delta_W->shape()[0] == in_dim && delta_W->shape()[1] == slice.cols) {
            // 加到指定列范围
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
    
    // Subtract LoRA deltas from base
    for (const auto& slice : slices_) {
        auto delta_W = matmul(slice.A, slice.B);
        delta_W = mul(delta_W, slice.scale);
        
        float* W_data = W_->data<float>();
        const float* delta_data = delta_W->data<float>();
        
        int64_t in_dim = W_->shape()[0];
        int64_t out_dim = W_->shape()[1];
        
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

}  // namespace ops
