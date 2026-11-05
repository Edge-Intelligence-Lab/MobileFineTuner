/**
 * @file lora_linear.cpp
 * @brief LoRA-augmented linear layer (manual LoRA-branch backward; A/B trainable)
 */

#include "../core/quant_ops.h"//Quantize
#include "lora_linear.h"
#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include "../core/backward_functions.h"
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace ops {

namespace {
float read_tensor_float(const TensorPtr& t, int64_t idx) {
    switch (t->dtype()) {
        case kFloat32:
            return t->data<float>()[idx];
        case kFloat16:
            return fp16_bits_to_float32(t->data<uint16_t>()[idx]);
        case kBFloat16:
            return bf16_bits_to_float32(t->data<uint16_t>()[idx]);
        default:
            throw std::runtime_error("LoRALinear: expected floating-point tensor storage");
    }
}

void write_tensor_float(const TensorPtr& t, int64_t idx, float value) {
    switch (t->dtype()) {
        case kFloat32:
            t->data<float>()[idx] = value;
            return;
        case kFloat16:
            t->data<uint16_t>()[idx] = float32_to_fp16_bits(value);
            return;
        case kBFloat16:
            t->data<uint16_t>()[idx] = float32_to_bf16_bits(value);
            return;
        default:
            throw std::runtime_error("LoRALinear: expected floating-point tensor storage");
    }
}

struct EffectiveLoraShape {
    int64_t in_dim = 0;
    int64_t rank = 0;
    int64_t out_dim = 0;
    bool a_transposed = false;
    bool b_transposed = false;
};

EffectiveLoraShape infer_effective_lora_shape(const TensorPtr& A,
                                              const TensorPtr& B,
                                              int64_t in_dim) {
    if (!A || !B || A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("LoRALinear: LoRA A/B must be 2D tensors");
    }

    EffectiveLoraShape shape;
    shape.in_dim = in_dim;
    if (A->shape()[0] == in_dim) {
        shape.rank = A->shape()[1];
        shape.a_transposed = false;
    } else if (A->shape()[1] == in_dim) {
        shape.rank = A->shape()[0];
        shape.a_transposed = true;
    } else {
        throw std::runtime_error("LoRALinear: LoRA A shape is incompatible with input dimension");
    }

    if (B->shape()[0] == shape.rank) {
        shape.out_dim = B->shape()[1];
        shape.b_transposed = false;
    } else if (B->shape()[1] == shape.rank) {
        shape.out_dim = B->shape()[0];
        shape.b_transposed = true;
    } else {
        throw std::runtime_error("LoRALinear: LoRA B shape is incompatible with rank");
    }

    return shape;
}

float lora_a_at(const TensorPtr& A, const EffectiveLoraShape& shape, int64_t in_idx, int64_t r_idx) {
    int64_t idx = shape.a_transposed
        ? r_idx * shape.in_dim + in_idx
        : in_idx * shape.rank + r_idx;
    return read_tensor_float(A, idx);
}

float lora_b_at(const TensorPtr& B, const EffectiveLoraShape& shape, int64_t r_idx, int64_t out_idx) {
    int64_t idx = shape.b_transposed
        ? out_idx * shape.rank + r_idx
        : r_idx * shape.out_dim + out_idx;
    return read_tensor_float(B, idx);
}

size_t tensor_nbytes(const TensorPtr& t) {
    return static_cast<size_t>(t->numel()) * DTypeUtils::size_of(t->dtype());
}

void apply_lora_delta_to_base(const TensorPtr& W, const LoRASlice& slice, float sign) {
    if (!DTypeUtils::is_floating_point(W->dtype())) {
        throw std::runtime_error("LoRALinear: merge requires floating-point base weights");
    }
    if (W->ndim() != 2) {
        throw std::runtime_error("LoRALinear: merge requires a 2D base weight");
    }

    const int64_t in_dim = W->shape()[0];
    const int64_t out_dim = W->shape()[1];
    const auto view = infer_effective_lora_shape(slice.A, slice.B, in_dim);
    const int64_t expected_cols = slice.cols > 0 ? slice.cols : view.out_dim;
    if (expected_cols != view.out_dim) {
        throw std::runtime_error("LoRALinear: LoRA slice cols do not match B output dimension");
    }
    if (slice.col0 < 0 || slice.col0 + view.out_dim > out_dim) {
        throw std::runtime_error("LoRALinear: LoRA slice range exceeds base weight width");
    }

    for (int64_t i = 0; i < in_dim; ++i) {
        for (int64_t o = 0; o < view.out_dim; ++o) {
            double delta = 0.0;
            for (int64_t r = 0; r < view.rank; ++r) {
                delta += static_cast<double>(lora_a_at(slice.A, view, i, r)) *
                         static_cast<double>(lora_b_at(slice.B, view, r, o));
            }
            const int64_t w_idx = i * out_dim + (slice.col0 + o);
            const float updated = read_tensor_float(W, w_idx) +
                                  sign * slice.scale * static_cast<float>(delta);
            write_tensor_float(W, w_idx, updated);
        }
    }
}

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

// ---------------------------------------------------------------------------
// Backward for the FUSED quantized base matmul (y = x @ Wq^T).
//
// The base weight is frozen, so dL/dWq is not needed. The only gradient that
// must propagate is dL/dx = grad_output @ Wq   (shape [N, M] @ [M, K] -> [N, K]),
// computed by matmul_q8_0_rhs_T directly from the int8 blocks -- no fp32 copy
// of the weight is ever materialised.
// ---------------------------------------------------------------------------
struct QuantBaseMatmulBackward : public BackwardFunction {
    TensorPtr x_;
    const std::vector<block_q8_0>* wq_;
    int64_t m_, k_;

    QuantBaseMatmulBackward(const TensorPtr& x,
                            const std::vector<block_q8_0>* wq,
                            int64_t m, int64_t k)
        : x_(x), wq_(wq), m_(m), k_(k) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        if (k_ % Q8_0_BLOCK_SIZE != 0) {
            throw std::runtime_error(
                "QuantBaseMatmulBackward: K must be a multiple of 32");
        }

        const int64_t n = grad_output->numel() / m_;
        const float* g = grad_output->data<float>();

        auto grad_x = zeros({n, k_}, DType::kFloat32, grad_output->device());
        float* gx = grad_x->data<float>();

        matmul_q8_0_rhs_T(g, wq_->data(), gx, n, m_, k_);

        return {grad_x};
    }
};
}  // namespace


bool LoRALinear::quantize_base_weight_q8_0(bool allow_skip) {
    if (!W_) {
        throw std::runtime_error("LoRALinear::quantize_base_weight_q8_0: W_ is null");
    }
    if (W_->ndim() != 2) {
        throw std::runtime_error("quantize_base_weight_q8_0: W_ must be 2D");
    }

    const int64_t K = W_->shape()[0];   // in_features
    const int64_t M = W_->shape()[1];   // out_features
    const int64_t n = K * M;

    // The fused kernel keeps the quantized buffer row-major [M, K] so that, for
    // a fixed out-feature j, the whole K row is contiguous. Caching the
    // transposed copy therefore requires K to be a whole number of blocks.
    //
    // KNOWN LIMITATION: Q8_0 is a block format (32 elements per block) and this
    // implementation does no padding, so a layer whose in_features is not a
    // multiple of 32 cannot be represented. Real checkpoints do contain such
    // layers. Rather than aborting a demo, allow_skip leaves the layer in fp32.
    if (n % Q8_0_BLOCK_SIZE != 0 || K % Q8_0_BLOCK_SIZE != 0) {
        const std::string msg =
            "LoRALinear: skipping Q8_0 quantization for weight [" +
            std::to_string(K) + ", " + std::to_string(M) +
            "] -- in_features is not a multiple of 32 (Q8_0 block size). "
            "Layer stays fp32.";
        if (!allow_skip) {
            throw std::runtime_error(msg);
        }
        std::cerr << "[WARN] " << msg << std::endl;
        is_quantized_ = false;
        quantized_weight_.clear();
        quantized_numel_ = 0;
        return false;
    }

    quantized_numel_ = n;
    quantized_weight_.resize(static_cast<size_t>(n / Q8_0_BLOCK_SIZE));

    // W_ is stored [K, M]; the Q8_0 buffer we build is [M, K] (transposed).
    // Transposing here is a one-off O(K*M) layout change and lets every
    // subsequent forward read a contiguous K row per output feature.
    const float* w_data = W_->data<float>();
    std::vector<float> w_t(static_cast<size_t>(n));
    for (int64_t kk = 0; kk < K; ++kk) {
        const float* src_row = w_data + kk * M;
        for (int64_t j = 0; j < M; ++j) {
            w_t[static_cast<size_t>(j) * K + kk] = src_row[j];
        }
    }

    quantize_row_q8_0_ref(w_t.data(), quantized_weight_.data(), static_cast<int>(n));

    is_quantized_ = true;
    return true;
}

void LoRALinear::attach_lora(const TensorPtr& A, const TensorPtr& B,
                             float scale, int col0, int cols) {
    if (!A || !B) {
        throw std::runtime_error("LoRALinear::attach_lora: A and B must not be null");
    }
    
    // Shape checks
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("LoRALinear::attach_lora: A and B must be 2D");
    }
    
    int64_t out_cols = cols;
    if (out_cols <= 0) {
        int64_t base_in_dim = W_->shape().empty() ? A->shape()[0] : W_->shape()[0];
        out_cols = infer_effective_lora_shape(A, B, base_in_dim).out_dim;
    }
    
    // LoRA parameters require gradients
    A->set_requires_grad(true);
    B->set_requires_grad(true);
    
    slices_.emplace_back(A, B, scale, col0, out_cols);
}

void LoRALinear::clear_lora() {
    if (merged_) {
        unmerge_from_base();
    }
    slices_.clear();
    merged_ = false;
    merge_backup_.clear();
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
    // Base weight tensor for the generic path (unquantized layers only).
    TensorPtr W_eff = W_;

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

    // Base branch: y = x @ W + b
    //
    // Paths:
    //   * fused (default for quantized layers) -- reads the int8 Q8_0 blocks
    //     directly, no fp32 copy of the weight is ever materialised.
    //   * dequantise-then-matmul -- used only as a fallback when the layer is
    //     quantized but the fused kernel cannot run (K not a multiple of 32).
    //
    // NOTE: an unquantized layer with in_features not a multiple of 32 simply
    // stays on the fp32 path -- quantize_base_weight_q8_0() refuses to mark such
    // a layer as quantized (see its allow_skip handling).
    const bool can_use_fused =
        is_quantized_ && (W_->ndim() == 2) &&
        (W_->shape()[0] % Q8_0_BLOCK_SIZE == 0);

    TensorPtr y;
    if (can_use_fused) {
        // W_ is stored [K, M]; the Q8_0 buffer is the transposed [M, K] layout,
        // so y2d[i][j] = dot(x_row_i, q8_row_j).
        const int64_t k = W_->shape()[0];   // in_features
        const int64_t m = W_->shape()[1];   // out_features

        const auto& x_shape = x->shape();
        const int64_t n = x->numel() / k;
        const int64_t kb = k / Q8_0_BLOCK_SIZE;

        // Optional instrumentation: attributes time to the activation
        // quantization, the kernel, and the output-buffer allocation.
        static const bool prof = std::getenv("QLORA_PROFILE_FUSED") != nullptr;
        static double p_quant_ms = 0.0, p_kernel_ms = 0.0, p_alloc_ms = 0.0;
        static double p_deq_ms = 0.0, p_fp32mm_ms = 0.0;
        static int64_t p_calls = 0;
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr, "[QLoRA-fused] FIRST CALL: prof=%d n=%lld k=%lld m=%lld\n",
                         (int)prof, (long long)n, (long long)k, (long long)m);
        }

        auto pg_now = [] {
            using clk = std::chrono::steady_clock;
            return std::chrono::duration<double, std::milli>(
                       clk::now().time_since_epoch()).count();
        };
        const double pg_t0 = prof ? pg_now() : 0.0;

        // Quantize the ACTIVATIONS as well. This buffer is scratch: it is
        // rebuilt on every forward call and never stored. Its only purpose is
        // to make both dot-product operands int8, so the inner loop is integer
        // arithmetic instead of float.
        std::vector<block_q8_0> x_q8(static_cast<size_t>(n * kb));
        const double pg_t1 = prof ? pg_now() : 0.0;
        quantize_activations_q8_0(x->data<float>(), x_q8.data(), n, k);

        const double pg_t2 = prof ? pg_now() : 0.0;
        auto y2d = zeros({n, m}, DType::kFloat32, x->device());

        const double pg_t3 = prof ? pg_now() : 0.0;
        static const char* tile_env = std::getenv("QLORA_TILE_ROWS");
        const int64_t tile_rows = tile_env ? std::atoll(tile_env) : 128;
        matmul_q8_0_q8_0_tiled(x_q8.data(), quantized_weight_.data(),
                               y2d->data<float>(), n, m, k, tile_rows);
        const double pg_t4 = prof ? pg_now() : 0.0;

        if (prof) {
            p_calls++;
            p_alloc_ms += (pg_t1 - pg_t0) + (pg_t3 - pg_t2);
            p_quant_ms += (pg_t2 - pg_t1);
            p_kernel_ms += (pg_t4 - pg_t3);

            // A/B on the SAME data and shapes: the fused int8 kernel versus a
            // dequantize-then-fp32 matmul. This is the only fair way to answer
            // "is the integer path actually faster here?".
            const double ta0 = pg_now();
            std::vector<float> w_deq(static_cast<size_t>(m) * k);
            dequantize_row_q8_0(quantized_weight_.data(), w_deq.data(),
                                static_cast<int>(m * k));
            auto w_eff = std::make_shared<Tensor>(
                std::vector<int64_t>{m, k}, w_deq.data(), kFloat32, kCPU, true);
            const double ta1 = pg_now();
            auto y_ref = matmul(x, w_eff);
            const double ta2 = pg_now();
            p_deq_ms += (ta1 - ta0);
            p_fp32mm_ms += (ta2 - ta1);

            if (p_calls <= 3 || (p_calls % 500) == 0) {
                std::fprintf(stderr,
                    "[QLoRA-profile] call=%lld n=%lld k=%lld m=%lld | "
                    "int8_kernel=%.2fms  dequant=%.2fms  fp32_matmul=%.2fms | "
                    "cum kernel=%.1f quant=%.1f deq=%.1f fp32mm=%.1f ms\n",
                    (long long)p_calls, (long long)n, (long long)k, (long long)m,
                    pg_t4 - pg_t3, ta1 - ta0, ta2 - ta1,
                    p_kernel_ms, p_quant_ms, p_deq_ms, p_fp32mm_ms);
            }
        } else if (std::getenv("QLORA_PROFILE_FUSED") != nullptr) {
            // env var set but this static was initialised earlier without it
            std::fprintf(stderr, "[QLoRA-profile] WARNING: profiling flag not latched\n");
        }

        // Propagate to x only when x actually needs a gradient.
        // set_grad_fn is honoured by both the new autograd engine and the
        // legacy path (see autograd_engine.cpp), and needs no access to the
        // engine-internal register_backward().
        if (x->requires_grad()) {
            y2d->set_requires_grad(true);
            auto backward_fn =
                std::make_shared<QuantBaseMatmulBackward>(x, &quantized_weight_, m, k);
            // Register the node with the engine so the dL/dx it returns is
            // actually routed to x.
            //
            // set_grad_fn() is NOT enough here: a tensor built by zeros() has no
            // input edges, so the engine takes the legacy branch (calls the
            // lambda once and ignores what it returns). The result was that the
            // frozen-weight path contributed NOTHING to dL/dx -- only the LoRA
            // branch did -- which silently degraded every quantized training run.
            ops::autograd::Engine::instance().register_node(y2d, {x}, backward_fn);
        }

        // Restore the leading batch dims, replacing the last (in-feature) axis
        // with the out-feature count -- NOT with x's last dim.
        std::vector<int64_t> out_shape = x_shape;
        out_shape.back() = m;
        y = reshape(y2d, out_shape);
    } else {
        // Unquantized base weight: the generic matmul path.
        log_shape("base matmul", x, W_eff);
        try {
            y = matmul(x, W_eff);
        } catch (const std::exception& e) {
            if (debug) {
                std::cerr << "[LoRALinear] base matmul threw: " << e.what() << std::endl;
            }
            throw;
        }
    }
    if (b_) {
        y = add(y, b_);
    }
    
    // LoRA delta: Σ scale * (x @ A @ B)
    if (!slices_.empty() && !merged_) {
        const int64_t w0 = W_->shape()[0];
        const int64_t w1 = W_->shape()[1];
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

    const size_t nbytes = tensor_nbytes(W_);
    merge_backup_.resize(nbytes);
    if (nbytes != 0) {
        std::memcpy(merge_backup_.data(), W_->data_ptr(), nbytes);
    }

    for (const auto& slice : slices_) {
        apply_lora_delta_to_base(W_, slice, +1.0f);
    }
    
    merged_ = true;
}

void LoRALinear::unmerge_from_base() {
    if (!merged_ || slices_.empty()) return;

    const size_t nbytes = tensor_nbytes(W_);
    if (!merge_backup_.empty()) {
        if (merge_backup_.size() != nbytes) {
            throw std::runtime_error("LoRALinear::unmerge_from_base: merge backup size mismatch");
        }
        if (nbytes != 0) {
            std::memcpy(W_->data_ptr(), merge_backup_.data(), nbytes);
        }
        merge_backup_.clear();
        merged_ = false;
        return;
    }

    for (const auto& slice : slices_) {
        apply_lora_delta_to_base(W_, slice, -1.0f);
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
