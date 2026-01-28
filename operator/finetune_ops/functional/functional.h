/**
 * @file functional.h
 * @brief Functional API for tensor operations
 * 
 * This file providess a functional interface to tensor operations,
 * similar to to PyTorch's functional API. It includes stateless functions
 * for mathematical operations, neural network layers, and utilities.
 */

#pragma once

#include "../core/tensor_unified.h"
#include <memory>

namespace ops {
/**
 * @brief Functional API namespace
 * 
 * Provides stateless functional operations for tensors.
 * These functions are designed to be used in a functional programming
 * style and can be easily composed.
 */
namespace functional {

// ============================================================================
// Linear Algebra Operations
// ============================================================================

/** @brief Matrix multiplication of two tensors */
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
/** @brief Additive matrix multiplication: bias + alpha * (a @ b) * beta */
TensorPtr addmm(const TensorPtr& bias, const TensorPtr& a, const TensorPtr& b, float alpha = 1.0f, float beta = 1.0f);

TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
TensorPtr div(const TensorPtr& a, const TensorPtr& b);

TensorPtr add(const TensorPtr& tensor, float scalar);
TensorPtr sub(const TensorPtr& tensor, float scalar);
TensorPtr mul(const TensorPtr& tensor, float scalar);
TensorPtr div(const TensorPtr& tensor, float scalar);

TensorPtr relu(const TensorPtr& input);
TensorPtr gelu(const TensorPtr& input);
TensorPtr sigmoid(const TensorPtr& input);
TensorPtr tanh(const TensorPtr& input);
TensorPtr softmax(const TensorPtr& input, int dim = -1);
TensorPtr log_softmax(const TensorPtr& input, int dim = -1);

TensorPtr layer_norm(const TensorPtr& input,
                    const std::vector<int64_t>& normalized_shape,
                    const TensorPtr& weight = nullptr,
                    const TensorPtr& bias = nullptr,
                    float eps = 1e-5);

TensorPtr batch_norm(const TensorPtr& input,
                    const TensorPtr& running_mean,
                    const TensorPtr& running_var,
                    const TensorPtr& weight = nullptr,
                    const TensorPtr& bias = nullptr,
                    bool training = true,
                    float momentum = 0.1,
                    float eps = 1e-5);

TensorPtr linear(const TensorPtr& input,
                const TensorPtr& weight,
                const TensorPtr& bias = nullptr);

TensorPtr embedding(const TensorPtr& input,
                   const TensorPtr& weight,
                   int padding_idx = -1,
                   float max_norm = -1,
                   float norm_type = 2.0,
                   bool scale_grad_by_freq = false,
                   bool sparse = false);

struct AttentionOutput {
    TensorPtr output;
    TensorPtr attention_weights;
};

AttentionOutput scaled_dot_product_attention(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    const TensorPtr& attn_mask = nullptr,
    float dropout_p = 0.0,
    bool is_causal = false
);

TensorPtr multi_head_attention(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    const TensorPtr& q_weight,
    const TensorPtr& k_weight,
    const TensorPtr& v_weight,
    const TensorPtr& out_weight,
    const TensorPtr& q_bias = nullptr,
    const TensorPtr& k_bias = nullptr,
    const TensorPtr& v_bias = nullptr,
    const TensorPtr& out_bias = nullptr,
    int num_heads = 8,
    const TensorPtr& attn_mask = nullptr,
    float dropout_p = 0.0,
    bool is_causal = false
);

TensorPtr cross_entropy_loss(const TensorPtr& input,
                            const TensorPtr& target,
                            const TensorPtr& weight = nullptr,
                            int ignore_index = -100,
                            const std::string& reduction = "mean");

TensorPtr mse_loss(const TensorPtr& input,
                  const TensorPtr& target,
                  const std::string& reduction = "mean");

TensorPtr nll_loss(const TensorPtr& input,
                  const TensorPtr& target,
                  const TensorPtr& weight = nullptr,
                  int ignore_index = -100,
                  const std::string& reduction = "mean");

TensorPtr conv1d(const TensorPtr& input,
                const TensorPtr& weight,
                const TensorPtr& bias = nullptr,
                int stride = 1,
                int padding = 0,
                int dilation = 1,
                int groups = 1);

TensorPtr conv2d(const TensorPtr& input,
                const TensorPtr& weight,
                const TensorPtr& bias = nullptr,
                const std::vector<int>& stride = {1, 1},
                const std::vector<int>& padding = {0, 0},
                const std::vector<int>& dilation = {1, 1},
                int groups = 1);

TensorPtr max_pool1d(const TensorPtr& input,
                    int kernel_size,
                    int stride = -1,
                    int padding = 0,
                    int dilation = 1,
                    bool ceil_mode = false);

TensorPtr avg_pool1d(const TensorPtr& input,
                    int kernel_size,
                    int stride = -1,
                    int padding = 0,
                    bool ceil_mode = false,
                    bool count_include_pad = true);

TensorPtr dropout(const TensorPtr& input, float p = 0.5, bool training = true);

TensorPtr reshape(const TensorPtr& input, const std::vector<int64_t>& shape);
TensorPtr view(const TensorPtr& input, const std::vector<int64_t>& shape);
TensorPtr transpose(const TensorPtr& input, int dim0, int dim1);
TensorPtr permute(const TensorPtr& input, const std::vector<int>& dims);
TensorPtr squeeze(const TensorPtr& input, int dim = -1);
TensorPtr unsqueeze(const TensorPtr& input, int dim);

TensorPtr cat(const std::vector<TensorPtr>& tensors, int dim = 0);
std::vector<TensorPtr> split(const TensorPtr& tensor, int split_size, int dim = 0);
std::vector<TensorPtr> chunk(const TensorPtr& tensor, int chunks, int dim = 0);

TensorPtr index_select(const TensorPtr& input, int dim, const TensorPtr& index);
TensorPtr gather(const TensorPtr& input, int dim, const TensorPtr& index);
TensorPtr scatter(const TensorPtr& input, int dim, const TensorPtr& index, const TensorPtr& src);

TensorPtr sum(const TensorPtr& input, int dim = -1, bool keepdim = false);
TensorPtr mean(const TensorPtr& input, int dim = -1, bool keepdim = false);
TensorPtr max(const TensorPtr& input, int dim = -1, bool keepdim = false);
TensorPtr min(const TensorPtr& input, int dim = -1, bool keepdim = false);

TensorPtr eq(const TensorPtr& a, const TensorPtr& b);
TensorPtr ne(const TensorPtr& a, const TensorPtr& b);
TensorPtr gt(const TensorPtr& a, const TensorPtr& b);
TensorPtr lt(const TensorPtr& a, const TensorPtr& b);
TensorPtr ge(const TensorPtr& a, const TensorPtr& b);
TensorPtr le(const TensorPtr& a, const TensorPtr& b);

TensorPtr logical_and(const TensorPtr& a, const TensorPtr& b);
TensorPtr logical_or(const TensorPtr& a, const TensorPtr& b);
TensorPtr logical_not(const TensorPtr& input);

TensorPtr sin(const TensorPtr& input);
TensorPtr cos(const TensorPtr& input);
TensorPtr tan(const TensorPtr& input);
TensorPtr asin(const TensorPtr& input);
TensorPtr acos(const TensorPtr& input);
TensorPtr atan(const TensorPtr& input);

TensorPtr exp(const TensorPtr& input);
TensorPtr log(const TensorPtr& input);
TensorPtr log2(const TensorPtr& input);
TensorPtr log10(const TensorPtr& input);
TensorPtr sqrt(const TensorPtr& input);
TensorPtr rsqrt(const TensorPtr& input);
TensorPtr pow(const TensorPtr& input, float exponent);

TensorPtr positional_encoding(int seq_len, int d_model, float max_freq = 10000.0);

TensorPtr causal_mask(int seq_len, DType dtype = kFloat32, Device device = kCPU);
TensorPtr attention_mask(const TensorPtr& input, int pad_token_id = 0);

}
}
