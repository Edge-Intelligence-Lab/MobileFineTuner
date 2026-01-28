/**
 * @file ops.h
 * @brief Core operations and functions for the operators framework
 * 
 * This file declares all the fundamental operations available in the framework,
 * including mathematical operations, neural network layers, and utility functions.
 * All operations support automatic differentiation and are designed to work
 * seamlessly with the Tensor class.
 */

#pragma once

#include "tensor.h"
#include <vector>
#include <memory>

namespace ops {

// ============================================================================
// Basic Arithmetic Operations
// ============================================================================

/** @brief Element-wise addition of two tensors */
TensorPtr add(const TensorPtr& a, const TensorPtr& b);
/** @brief Element-wise subtraction of two tensors */
TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
/** @brief Element-wise multiplication of two tensors */
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
/** @brief Element-wise division of two tensors */
TensorPtr div(const TensorPtr& a, const TensorPtr& b);

// Scalar operations (tensor + scalar)
/** @brief Add scalar to tensor element-wise */
TensorPtr add(const TensorPtr& tensor, float scalar);
/** @brief Subtract scalar from tensor element-wise */
TensorPtr sub(const TensorPtr& tensor, float scalar);
/** @brief Multiply tensor by scalar element-wise */
TensorPtr mul(const TensorPtr& tensor, float scalar);
/** @brief Divide tensor by scalar element-wise */
TensorPtr div(const TensorPtr& tensor, float scalar);

// Scalar operations (scalar + tensor)
/** @brief Add tensor to scalar element-wise */
TensorPtr add(float scalar, const TensorPtr& tensor);
/** @brief Subtract tensor from scalar element-wise */
TensorPtr sub(float scalar, const TensorPtr& tensor);
/** @brief Multiply scalar by tensor element-wise */
TensorPtr mul(float scalar, const TensorPtr& tensor);
/** @brief Divide scalar by tensor element-wise */
TensorPtr div(float scalar, const TensorPtr& tensor);

// ============================================================================
// Linear Algebra Operations
// ============================================================================

/** @brief Matrix multiplication of two tensors */
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
/** @brief Matrix multiplication with right-hand-side transpose (zero-copy for tying)
 *  result = a @ b^T, b is transposed on-the-fly without allocating b^T */
TensorPtr matmul_rhs_T(const TensorPtr& a, const TensorPtr& b);
/** @brief Transpose two dimensions of a tensor */
TensorPtr transpose(const TensorPtr& tensor, int dim0, int dim1);
/** @brief Permute dimensions of a tensor according to given order */
TensorPtr permute(const TensorPtr& tensor, const std::vector<int>& dims);

/** @brief Linear transforation: output = input @ weight + bias */
TensorPtr linear(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias = nullptr);

// LoRA linear layer is declared later (line 117-119)

// ============================================================================
// Activation Functions
// ============================================================================

/** @brief ReLU activation: max(0, x) */
TensorPtr relu(const TensorPtr& x);
/** @brief GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
TensorPtr gelu(const TensorPtr& x);
/** @brief SiLU (Swish) activation: x * sigmoid(x) */
TensorPtr silu(const TensorPtr& x);
/** @brief Sigmoid activation: 1 / (1 + exp(-x)) */
TensorPtr sigmoid(const TensorPtr& x);
/** @brief Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) */
TensorPtr tanh_op(const TensorPtr& x);
/** @brief Alias for Tanh activation to match common API */
inline TensorPtr tanh(const TensorPtr& x) { return tanh_op(x); }
/** @brief Softmax activation: exp(x_i) / sum(exp(x_j)) for all j */
TensorPtr softmax(const TensorPtr& x, int dim = -1);
/** @brief Log-softmax activation: log(softmax(x)) */
TensorPtr log_softmax(const TensorPtr& x, int dim = -1);

// ============================================================================
// Loss Functions
// ============================================================================

/** @brief Mean Squared Error loss: mean((input - target)²) */
TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction = "mean");
TensorPtr cross_entropy_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction = "mean");
TensorPtr nll_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction = "mean");

TensorPtr layer_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias, float eps = 1e-5f);
TensorPtr rms_norm(const TensorPtr& input, const TensorPtr& weight, float eps = 1e-6f);
// Variant without "+1": y = x_hat * weight
TensorPtr rms_norm_affine(const TensorPtr& input, const TensorPtr& weight, float eps = 1e-6f);
TensorPtr batch_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
                     const TensorPtr& running_mean = nullptr, const TensorPtr& running_var = nullptr,
                     bool training = true, float momentum = 0.1f, float eps = 1e-5f);

TensorPtr lora_linear(const TensorPtr& input, const TensorPtr& weight, 
                     const TensorPtr& lora_A, const TensorPtr& lora_B, 
                     float alpha = 1.0f, const TensorPtr& bias = nullptr);

TensorPtr reshape(const TensorPtr& tensor, const std::vector<int64_t>& shape);
TensorPtr view(const TensorPtr& tensor, const std::vector<int64_t>& shape);

// ============================================================================
// Data Type Operations (Mixed Precision Training)
// ============================================================================

/** @brief Cast tensor to different data type (FP32 <-> FP16 conversion) */
TensorPtr cast(const TensorPtr& tensor, DType target_dtype);

/** @brief Convert tensor to FP16 (alias for cast) */
inline TensorPtr to_fp16(const TensorPtr& tensor) { return cast(tensor, DType::kFloat16); }

/** @brief Convert tensor to FP32 (alias for cast) */
inline TensorPtr to_fp32(const TensorPtr& tensor) { return cast(tensor, DType::kFloat32); }
TensorPtr flatten(const TensorPtr& tensor, int start_dim = 0, int end_dim = -1);
TensorPtr squeeze(const TensorPtr& tensor, int dim = -1);
TensorPtr unsqueeze(const TensorPtr& tensor, int dim);

// TODO: 以下函数声明但未实现，如需使用请补充实现
// TensorPtr index_select(const TensorPtr& tensor, int dim, const TensorPtr& index);
// TensorPtr masked_select(const TensorPtr& tensor, const TensorPtr& mask);
// TensorPtr where(const TensorPtr& condition, const TensorPtr& x, const TensorPtr& y);
// TensorPtr gather(const TensorPtr& tensor, int dim, const TensorPtr& index);
// TensorPtr scatter(const TensorPtr& tensor, int dim, const TensorPtr& index, const TensorPtr& src);

TensorPtr sum(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
TensorPtr mean(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
// TODO: 以下函数声明但未实现，如需使用请补充实现
// TensorPtr max(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
// TensorPtr min(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
// TODO: 以下函数声明但未实现，如需使用请补充实现
// TensorPtr argmax(const TensorPtr& tensor, int dim = -1, bool keepdim = false);
// TensorPtr argmin(const TensorPtr& tensor, int dim = -1, bool keepdim = false);

TensorPtr eq(const TensorPtr& a, const TensorPtr& b);
TensorPtr ne(const TensorPtr& a, const TensorPtr& b);
TensorPtr gt(const TensorPtr& a, const TensorPtr& b);
TensorPtr lt(const TensorPtr& a, const TensorPtr& b);
TensorPtr ge(const TensorPtr& a, const TensorPtr& b);
TensorPtr le(const TensorPtr& a, const TensorPtr& b);

TensorPtr dropout(const TensorPtr& tensor, float p = 0.5f, bool training = true);
// TODO: normal is not implemented; use randn instead
// TensorPtr normal(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f,
//                  DType dtype = kFloat32, Device device = kCPU);
TensorPtr uniform(const std::vector<int64_t>& shape, float low = 0.0f, float high = 1.0f,
                  DType dtype = kFloat32, Device device = kCPU);
inline TensorPtr unifor(const std::vector<int64_t>& shape, float low = 0.0f, float high = 1.0f,
                        DType dtype = kFloat32, Device device = kCPU) {
    return uniform(shape, low, high, dtype, device);
}

TensorPtr abs(const TensorPtr& tensor);
TensorPtr sqrt(const TensorPtr& tensor);
TensorPtr exp(const TensorPtr& tensor);
TensorPtr log(const TensorPtr& tensor);
TensorPtr pow(const TensorPtr& tensor, float exponent);
TensorPtr clamp(const TensorPtr& tensor, float min_val, float max_val);

// Comparison operators (implemented)

TensorPtr create_causal_mask(int seq_len, DType dtype = kFloat32, Device device = kCPU);
TensorPtr apply_mask(const TensorPtr& input, const TensorPtr& mask, float mask_value = -1e9f);
TensorPtr repeat_kv_heads(const TensorPtr& kv, int repeat_factor);
TensorPtr apply_rope(const TensorPtr& x, int seq_len, int head_dim, float rope_theta = 10000.0f);
TensorPtr swiglu(const TensorPtr& gate, const TensorPtr& up);

bool same_shape(const TensorPtr& a, const TensorPtr& b);
bool broadcastable(const TensorPtr& a, const TensorPtr& b);
std::vector<int64_t> broadcast_shape(const TensorPtr& a, const TensorPtr& b);

// TODO: to_float/to_int are not implemented; use cast instead
// TensorPtr to_float(const TensorPtr& tensor);
// TensorPtr to_int(const TensorPtr& tensor);

std::vector<int64_t> infer_broadcast_shape(const TensorPtr& a, const TensorPtr& b);

}
