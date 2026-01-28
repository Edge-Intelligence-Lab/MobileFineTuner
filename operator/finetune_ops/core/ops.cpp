/**
 * @file ops.cpp
 * @brief Implementation of core operations for the operators framework
 * 
 * This file contains the implementsation of all mathematical operations,
 * neural network layers, and utility functions declared in ops.h.
 * All operations support automatic differentiation and are optimized
 * for both CPU and potential GPU execution.
 */

#include "ops.h"
#include "backward_functions.h"
#include "autograd_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

// 默认使用纯 C++，当启用 USE_BLAS 时走 BLAS 实现

namespace ops {

// Helper: register node with new engine (if enabled) or fallback to legacy
namespace {
    void register_backward(const TensorPtr& output,
                          const std::vector<TensorPtr>& inputs,
                          BackwardFunctionPtr backward_fn) {
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        try {
            autograd::Engine::instance().register_node(output, inputs, backward_fn);
        } catch (const std::exception& e) {
            std::cerr << "[register_backward] Exception: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[register_backward] Unknown exception" << std::endl;
            throw;
        }
        #else
        // Legacy: set grad_fn that calls accumulate_gradient
        // (kept for backward compatibility)
        #endif
    }
}

namespace {

    inline uint16_t float32_to_fp16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;

        if (exponent <= 0) {
            if (exponent < -10) {
                return static_cast<uint16_t>(sign);
            }
            mantissa |= 0x800000u;
            uint32_t shifted = mantissa >> (1 - exponent + 13);
            return static_cast<uint16_t>(sign | shifted);
        }

        if (exponent >= 31) {
            uint16_t inf_nan = (mantissa == 0) ? 0x7C00u : static_cast<uint16_t>(0x7C00u | (mantissa >> 13));
            return static_cast<uint16_t>(sign | inf_nan);
        }

        uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
        return half;
    }

    inline float fp16_to_float32(uint16_t value) {
        uint32_t sign = (value & 0x8000u) << 16;
        uint32_t exponent = (value >> 10) & 0x1Fu;
        uint32_t mantissa = value & 0x3FFu;

        uint32_t bits = 0;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                // subnormal
                exponent = 1;
                while ((mantissa & 0x400u) == 0) {
                    mantissa <<= 1;
                    --exponent;
                }
                mantissa &= 0x3FFu;
                exponent = exponent - 1 + 127 - 15;
                bits = sign | (exponent << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            bits = sign | 0x7F800000u | (mantissa << 13);
        } else {
            exponent = exponent - 15 + 127;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    /**
     * @brief Check if two tensors have the same shape
     * @param a First tensor
     * @param b Second tensor
     * @return True if shapes are equal
     */
    bool shapes_equal(const TensorPtr& a, const TensorPtr& b) {
        return a->shape() == b->shape();
    }

    /**
     * @brief Check if two tensors can be broadcast together
     * @param a First tensor
     * @param b Second tensor
     * @return True if tensors can be broadcast
     */
    bool can_broadcast(const TensorPtr& a, const TensorPtr& b) {
        const auto& shape_a = a->shape();
        const auto& shape_b = b->shape();

        int max_ndim = std::max(shape_a.size(), shape_b.size());

        for (int i = 0; i < max_ndim; ++i) {
            int dim_a = (static_cast<size_t>(i) < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
            int dim_b = (static_cast<size_t>(i) < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }

        return true;
    }

    std::vector<int64_t> broadcast_shapes(const TensorPtr& a, const TensorPtr& b) {
        const auto& shape_a = a->shape();
        const auto& shape_b = b->shape();

        int max_ndim = std::max(shape_a.size(), shape_b.size());
        std::vector<int64_t> result_shape(max_ndim);

        for (int i = 0; i < max_ndim; ++i) {
            int64_t dim_a = (static_cast<size_t>(i) < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
            int64_t dim_b = (static_cast<size_t>(i) < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

            result_shape[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        }

        return result_shape;
    }

    template<typename Op>
    TensorPtr elementwise_binary_op(const TensorPtr& a, const TensorPtr& b, Op op) {
        if (!can_broadcast(a, b)) {
            // 🔧 添加详细错误信息
            std::string msg = "Tensors cannot be broadcasted: shape_a=[";
            for (size_t i = 0; i < a->shape().size(); ++i) {
                msg += std::to_string(a->shape()[i]);
                if (i < a->shape().size() - 1) msg += ",";
            }
            msg += "], shape_b=[";
            for (size_t i = 0; i < b->shape().size(); ++i) {
                msg += std::to_string(b->shape()[i]);
                if (i < b->shape().size() - 1) msg += ",";
            }
            msg += "]";
            throw TensorError(msg);
        }

        auto result_shape = broadcast_shapes(a, b);
        auto result = zeros(result_shape, a->dtype(), a->device());

        if (shapes_equal(a, b)) {
            const float* data_a = a->data<float>();
            const float* data_b = b->data<float>();
            float* result_data = result->data<float>();

            for (int64_t i = 0; i < a->numel(); ++i) {
                result_data[i] = op(data_a[i], data_b[i]);
            }
        } else {
            // Complete broadcast implementsation
            const float* data_a = a->data<float>();
            const float* data_b = b->data<float>();
            float* result_data = result->data<float>();
            
            auto shape_a = a->shape();
            auto shape_b = b->shape();
            
            for (int64_t i = 0; i < result->numel(); ++i) {
                // Calculate multidimensional index of current position in result
                std::vector<int64_t> result_idx(result_shape.size());
                int64_t temp = i;
                for (int j = result_shape.size() - 1; j >= 0; --j) {
                    result_idx[j] = temp % result_shape[j];
                    temp /= result_shape[j];
                }
                
                // Calculate corresponding indices for a and b (simplified version)
                int64_t idx_a = 0, idx_b = 0;
                
                // Calculate linear index for a
                for (size_t dim = 0; dim < shape_a.size(); ++dim) {
                    int result_dim = dim + (result_shape.size() - shape_a.size());
                    if (result_dim >= 0) {
                        int64_t coord = (shape_a[dim] == 1) ? 0 : result_idx[result_dim];
                        idx_a = idx_a * shape_a[dim] + coord;
                    }
                }
                
                // Calculate linear index for b
                for (size_t dim = 0; dim < shape_b.size(); ++dim) {
                    int result_dim = dim + (result_shape.size() - shape_b.size());
                    if (result_dim >= 0) {
                        int64_t coord = (shape_b[dim] == 1) ? 0 : result_idx[result_dim];
                        idx_b = idx_b * shape_b[dim] + coord;
                    }
                }
                
                result_data[i] = op(data_a[idx_a], data_b[idx_b]);
            }
        }

        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);

        }

        return result;
    }

    template<typename Op>
    TensorPtr elementwise_unary_op(const TensorPtr& x, Op op) {
        auto result = zeros(x->shape(), x->dtype(), x->device());

        const float* x_data = x->data<float>();
        float* result_data = result->data<float>();

        for (int64_t i = 0; i < x->numel(); ++i) {
            result_data[i] = op(x_data[i]);
        }

        if (x->requires_grad()) {
            result->set_requires_grad(true);

            result->set_grad_fn([x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {

                auto grad_input = zeros(x->shape(), x->dtype(), x->device());
                const float* grad_out_data = grad_output->data<float>();
                float* grad_in_data = grad_input->data<float>();

                for (int64_t i = 0; i < x->numel(); ++i) {
                    grad_in_data[i] = grad_out_data[i];
                }

                if (x->requires_grad()) {
                    accumulate_gradient(x, grad_input);
                }

                return {grad_input};
            });
        }

        return result;
    }
}

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x + y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<AddBackward>(a->shape(), b->shape());
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr sub(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x - y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<SubBackward>(a->shape(), b->shape());
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) { return x * y; });

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<MulBackward>(a, b);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr div(const TensorPtr& a, const TensorPtr& b) {
    auto result = elementwise_binary_op(a, b, [](float x, float y) {
        if (y == 0.0f) throw TensorError("Division by zero");
        return x / y;
    });
    
    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<DivBackward>(a, b);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (a->requires_grad()) accumulate_gradient(a, grads[0]);
            if (b->requires_grad()) accumulate_gradient(b, grads[1]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr add(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x + scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr sub(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x - scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<PassThroughBackward>();
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr mul(const TensorPtr& tensor, float scalar) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x * scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ScaleBackward>(scalar);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr div(const TensorPtr& tensor, float scalar) {
    if (scalar == 0.0f) throw TensorError("Division by zero");
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return x / scalar; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ScaleBackward>(1.0f / scalar);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr add(float scalar, const TensorPtr& tensor) {
    return add(tensor, scalar);
}

TensorPtr sub(float scalar, const TensorPtr& tensor) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) { return scalar - x; });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // d(scalar - x)/dx = -1
        auto backward_fn = std::make_shared<ScaleBackward>(-1.0f);
        register_backward(result, {tensor}, backward_fn);
        #endif
    }
    return result;
}

TensorPtr mul(float scalar, const TensorPtr& tensor) {
    auto result = mul(tensor, scalar);
    // mul(tensor, scalar) already handles registration
    return result;
}

TensorPtr div(float scalar, const TensorPtr& tensor) {
    auto result = elementwise_unary_op(tensor, [scalar](float x) {
        if (x == 0.0f) throw TensorError("Division by zero");
        return scalar / x;
    });
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<ScalarOverTensorBackward>(tensor, scalar);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    return result;
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw TensorError("matmul: input tensors must not be null");
    }
    
    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();

    #ifdef AUTOGRAD_DEBUG
    // [Translated comment removed - see documentation]
    #endif
    
    if (shape_a.size() < 2 || shape_b.size() < 2) {
        throw TensorError("matmul requires tensors with at least 2 dimensions");
    }

    int64_t m = shape_a[shape_a.size() - 2];
    int64_t k = shape_a[shape_a.size() - 1];
    int64_t n = shape_b[shape_b.size() - 1];

    if (k != shape_b[shape_b.size() - 2]) {
        // 🔧 Auto-correct: common case is B shaped [Out, In]; if A's K matches B's last dim, transpose B dynamically
        if (shape_b.size() >= 2 && k == shape_b[shape_b.size() - 1]) {
            // Dynamically transpose the last two dims of B
            auto b_t = transpose(b, (int)shape_b.size() - 2, (int)shape_b.size() - 1);
            return matmul(a, b_t);
        }

        if (std::getenv("OPS_DEBUG_MATMUL_MISMATCH")) {
            std::cerr << "[matmul debug] mismatch A_ptr=" << a.get()
                      << " B_ptr=" << b.get() << std::endl;
        }

        // Detailed error message
        std::string error_msg = "matmul dimension mismatch:\n";
        error_msg += "  A shape: [";
        for (size_t i = 0; i < shape_a.size(); ++i) {
            error_msg += std::to_string(shape_a[i]);
            if (i < shape_a.size() - 1) error_msg += ", ";
        }
        error_msg += "]\n  B shape: [";
        for (size_t i = 0; i < shape_b.size(); ++i) {
            error_msg += std::to_string(shape_b[i]);
            if (i < shape_b.size() - 1) error_msg += ", ";
        }
        error_msg += "]\n  A K dimension=" + std::to_string(k);
        error_msg += ", B K dimension=" + std::to_string(shape_b[shape_b.size() - 2]);
        error_msg += "\n  💡 Hint: if B is a weight matrix, transpose(B) may be needed";
        error_msg += "\n  💡 If A=[1,63,640] and B=[256,640], a grad propagation path is likely incorrect";
        throw TensorError(error_msg);
    }

    auto result_shape = shape_a;
    result_shape[result_shape.size() - 2] = m;
    result_shape[result_shape.size() - 1] = n;

    auto result = zeros(result_shape, a->dtype(), a->device());

    // Pure C++ matmul implementation
    auto naive_matmul = [](const float* A, const float* B, float* C, int64_t m, int64_t n, int64_t k) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    };
    
    if (shape_a.size() == 2 && shape_b.size() == 2) {
        const float* data_a = a->data<float>();
        const float* data_b = b->data<float>();
        float* result_data = result->data<float>();
        #ifdef USE_BLAS
        // Row-major GEMM: C[m,n] = A[m,k] * B[k,n]
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    1.0f,
                    data_a, (int)k,
                    data_b, (int)n,
                    0.0f,
                    result_data, (int)n);
        #else
        naive_matmul(data_a, data_b, result_data, m, n, k);
        #endif
    } else {
        // 🔧 Fix: handle all leading batch dimensions, not just the first
        // Compute total batches: shape[0] * shape[1] * ... * shape[ndim-2]
        int64_t total_batches = 1;
        for (size_t i = 0; i < shape_a.size() - 2; ++i) {
            total_batches *= shape_a[i];
        }
        
        int64_t a_rows = shape_a[shape_a.size() - 2];
        int64_t a_cols = shape_a[shape_a.size() - 1];
        int64_t b_cols = shape_b[shape_b.size() - 1];
        
        // Matrix sizes per batch
        int64_t matrix_size_a = a_rows * a_cols;
        int64_t matrix_size_b = a_cols * b_cols;
        int64_t matrix_size_result = a_rows * b_cols;

        const float* data_a = a->data<float>();
        const float* data_b = b->data<float>();
        float* result_data = result->data<float>();

        // Check whether B carries the same batch dimensions
        bool b_has_batch = (shape_b.size() == shape_a.size());
        if (b_has_batch) {
            for (size_t i = 0; i < shape_a.size() - 2; ++i) {
                if (shape_b[i] != shape_a[i]) {
                    b_has_batch = false;
                    break;
                }
            }
        }
        
        for (int64_t batch = 0; batch < total_batches; ++batch) {
            const float* batch_a = data_a + batch * matrix_size_a;
            const float* batch_b = b_has_batch ? 
                                  data_b + batch * matrix_size_b : 
                                  data_b;
            float* batch_result = result_data + batch * matrix_size_result;
            #ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans, CblasNoTrans,
                        (int)a_rows, (int)b_cols, (int)a_cols,
                        1.0f,
                        batch_a, (int)a_cols,
                        batch_b, (int)b_cols,
                        0.0f,
                        batch_result, (int)b_cols);
            #else
            naive_matmul(batch_a, batch_b, batch_result, a_rows, b_cols, a_cols);
            #endif
        }
    }

    if (a->requires_grad() || b->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<MatmulBackward>(a, b);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {a, b}, backward_fn);
        #else
        // Legacy recursive path
        result->set_grad_fn([backward_fn, a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (a->requires_grad()) {
                accumulate_gradient(a, grads[0]);
            }
            if (b->requires_grad()) {
                accumulate_gradient(b, grads[1]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr matmul_rhs_T(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) {
        throw TensorError("matmul_rhs_T: input tensors must not be null");
    }
    
    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();
    
    // b mustis 2D: [N, K]
    if (shape_b.size() != 2) {
        throw TensorError("matmul_rhs_T: b must be 2D [N, K]");
    }
    
    // a canis 2D or 3D
    if (shape_a.size() < 2) {
        throw TensorError("matmul_rhs_T: a must be at least 2D");
    }
    
    int64_t n = shape_b[0];
    int64_t k_b = shape_b[1];
    
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    
    if (shape_a.size() == 2) {
        // 2D: a[M,K] @ b[N,K]^T = result[M,N]
        int64_t m = shape_a[0];
        int64_t k_a = shape_a[1];
        
        if (k_a != k_b) {
            throw TensorError("matmul_rhs_T dimension mismatch: A has K=" + std::to_string(k_a) +
                             " but B has K=" + std::to_string(k_b));
        }
        
        auto result = zeros({m, n}, a->dtype(), a->device());
        float* result_data = result->data<float>();
        
        #ifdef USE_BLAS
        // Row-major: C[m,n] = A[m,k] * B[n,k]^T  => NoTrans x Trans
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasTrans,
                    (int)m, (int)n, (int)k_a,
                    1.0f,
                    data_a, (int)k_a,
                    data_b, (int)k_b,
                    0.0f,
                    result_data, (int)n);
        #else
        // Pure C++ matmul_rhs_T: C[m,n] = A[m,k] @ B[n,k]^T
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k_a; ++p) {
                    sum += data_a[i * k_a + p] * data_b[j * k_b + p];
                }
                result_data[i * n + j] = sum;
            }
        }
        #endif
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);
            auto backward_fn = std::make_shared<MatmulRhsTBackward>(a, b);
            register_backward(result, {a, b}, backward_fn);
        }
        #else
        if (a->requires_grad()) {
            result->set_requires_grad(true);
            result->set_grad_fn([a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grad_a = matmul(grad_output, b);
                if (a->requires_grad()) {
                    accumulate_gradient(a, grad_a);
                }
                return {};
            });
        }
        #endif
        
        return result;
    } else {
        // 3D+: a[..., M, K] @ b[N, K]^T = result[..., M, N]
        // bybatchprocess
        auto result_shape = shape_a;
        result_shape[result_shape.size() - 1] = n;
        auto result = zeros(result_shape, a->dtype(), a->device());
        float* result_data = result->data<float>();
        
        int64_t m = shape_a[shape_a.size() - 2];
        int64_t k_a = shape_a[shape_a.size() - 1];
        
        if (k_a != k_b) {
            throw TensorError("matmul_rhs_T dimension mismatch");
        }
        
        // Computebatchsize
        int64_t batch_size = 1;
        for (size_t i = 0; i < shape_a.size() - 2; ++i) {
            batch_size *= shape_a[i];
        }
        
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* batch_a = data_a + batch * (m * k_a);
            float* batch_result = result_data + batch * (m * n);
            
            #ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans, CblasTrans,
                        (int)m, (int)n, (int)k_a,
                        1.0f,
                        batch_a, (int)k_a,
                        data_b, (int)k_b,
                        0.0f,
                        batch_result, (int)n);
            #else
            // Pure C++ batch matmul_rhs_T
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int64_t p = 0; p < k_a; ++p) {
                        sum += batch_a[i * k_a + p] * data_b[j * k_b + p];
                    }
                    batch_result[i * n + j] = sum;
                }
            }
            #endif
        }
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        if (a->requires_grad() || b->requires_grad()) {
            result->set_requires_grad(true);
            auto backward_fn = std::make_shared<MatmulRhsTBackward>(a, b);
            register_backward(result, {a, b}, backward_fn);
        }
        #else
        if (a->requires_grad()) {
            result->set_requires_grad(true);
            result->set_grad_fn([a, b](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                auto grad_a = matmul(grad_output, b);
                if (a->requires_grad()) {
                    accumulate_gradient(a, grad_a);
                }
                return {};
            });
        }
        #endif
        
        return result;
    }
}

TensorPtr transpose(const TensorPtr& tensor, int dim0, int dim1) {
    const auto& shape = tensor->shape();
    int ndim = shape.size();
    
        // [Translated]
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    
    // checkdimensionvalidity
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw TensorError("transpose: invalid dimensions");
    }
    
    // createnewshape
    std::vector<int64_t> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);
    
    // createresulttensor
    auto result = zeros(new_shape, tensor->dtype(), tensor->device());
    
    // executetranspose
    const float* src_data = tensor->data<float>();
    float* dst_data = result->data<float>();
    
        // [Translated]
    std::vector<int64_t> strides_src(ndim), strides_dst(ndim);
    
    // Computesourcetensorstrides
    strides_src[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides_src[i] = strides_src[i + 1] * shape[i + 1];
    }
    
    // Computetargettensorstrides
    strides_dst[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides_dst[i] = strides_dst[i + 1] * new_shape[i + 1];
    }
    
    // executetranspose
    int64_t total_elements = result->numel();
    std::vector<int64_t> indices(ndim);
    
    for (int64_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        // [Translated comment removed - see documentation]
        int64_t temp = linear_idx;
        for (int i = 0; i < ndim; ++i) {
            indices[i] = temp / strides_dst[i];
            temp %= strides_dst[i];
        }
        
        // swapdimension
        std::swap(indices[dim0], indices[dim1]);
        
                // [Translated]
        int64_t src_idx = 0;
        for (int i = 0; i < ndim; ++i) {
            src_idx += indices[i] * strides_src[i];
        }
        
        dst_data[linear_idx] = src_data[src_idx];
    }
    
    // settingsgradientpropagate
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<TransposeBackward>(dim0, dim1);
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor, dim0, dim1](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                        // [Translated]
            auto grad_input = transpose(grad_output, dim0, dim1);
            accumulate_gradient(tensor, grad_input);
            return {};
        });
        #endif
    }
    
    return result;
}

TensorPtr permute(const TensorPtr& tensor, const std::vector<int>& dims) {
    const auto& shape = tensor->shape();
    int ndim = static_cast<int>(shape.size());
    if (static_cast<int>(dims.size()) != ndim) {
        throw TensorError("permute: dims size must match tensor ndim");
    }
    // Validate dims as a permutation of [0..ndim-1]
    std::vector<int> seen(ndim, 0);
    for (int d : dims) {
        int dd = d < 0 ? d + ndim : d;
        if (dd < 0 || dd >= ndim) throw TensorError("permute: invalid dim index");
        if (seen[dd]) throw TensorError("permute: duplicate dim index");
        seen[dd] = 1;
    }

    // Build new shape
    std::vector<int64_t> new_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        int dd = dims[i] < 0 ? dims[i] + ndim : dims[i];
        new_shape[i] = shape[dd];
    }

    auto result = zeros(new_shape, tensor->dtype(), tensor->device());

    // Compute strides of source and destination
    std::vector<int64_t> src_strides(ndim), dst_strides(ndim);
    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) src_strides[i] = src_strides[i + 1] * shape[i + 1];
    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];

    const float* src = tensor->data<float>();
    float* dst = result->data<float>();

    int64_t total = result->numel();
    std::vector<int64_t> out_idx(ndim);
    std::vector<int64_t> in_idx(ndim);
    for (int64_t linear = 0; linear < total; ++linear) {
        // Decode dst index
        int64_t tmp = linear;
        for (int i = 0; i < ndim; ++i) {
            out_idx[i] = tmp / dst_strides[i];
            tmp %= dst_strides[i];
        }
        // Map to src index by inverse permutation
        for (int i = 0; i < ndim; ++i) {
            int dd = dims[i] < 0 ? dims[i] + ndim : dims[i];
            in_idx[dd] = out_idx[i];
        }
        // Encode src linear index
        int64_t src_lin = 0;
        for (int i = 0; i < ndim; ++i) src_lin += in_idx[i] * src_strides[i];
        dst[linear] = src[src_lin];
    }

    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        std::vector<int> inv(ndim);
        for (int i = 0; i < ndim; ++i) {
            int dd = dims[i] < 0 ? dims[i] + ndim : dims[i];
            inv[dd] = i;
        }
        auto backward_fn = std::make_shared<PermuteBackward>(inv);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr linear(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias) {

    auto result = matmul(input, transpose(weight, 0, 1));

    if (bias) {

        const auto& result_shape = result->shape();
        const auto& bias_shape = bias->shape();

        if (bias_shape.size() == 1 && result_shape.size() >= 1 &&
            bias_shape[0] == result_shape.back()) {

            auto broadcast_result = zeros(result_shape, result->dtype(), result->device());
            const float* result_data = result->data<float>();
            const float* bias_data = bias->data<float>();
            float* output_data = broadcast_result->data<float>();

            int64_t last_dim = result_shape.back();
            int64_t total_elements = result->numel();

            for (int64_t i = 0; i < total_elements; ++i) {
                int64_t bias_idx = i % last_dim;
                output_data[i] = result_data[i] + bias_data[bias_idx];
            }

            // Manual broadcast bias addition: Gradient propagation equivalent to add
            if (result->requires_grad() || bias->requires_grad()) {
                broadcast_result->set_requires_grad(true);
                #ifdef USE_NEW_AUTOGRAD_ENGINE
                auto backward_fn = std::make_shared<AddBackward>(result->shape(), bias->shape());
                register_backward(broadcast_result, {result, bias}, backward_fn);
                #endif
            }

            result = broadcast_result;
        } else {

            result = add(result, bias);
        }
    }

    if (input->requires_grad() || weight->requires_grad() || (bias && bias->requires_grad())) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<LinearBackward>(input, weight, bias);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {input, weight, bias}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, input, weight, bias](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (input->requires_grad()) {
                accumulate_gradient(input, grads[0]);
            }
            if (weight->requires_grad()) {
                accumulate_gradient(weight, grads[1]);
            }
            if (bias && bias->requires_grad()) {
                accumulate_gradient(bias, grads[2]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr relu(const TensorPtr& x) {
    auto result = elementwise_unary_op(x, [](float val) { return std::max(0.0f, val); });

    if (x->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<ReluBackward>(x);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            accumulate_gradient(x, grads[0]);
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr gelu(const TensorPtr& x) {
    auto result = elementwise_unary_op(x, [](float val) {
        float tanh_input = 0.7978845608f * (val + 0.044715f * val * val * val);
        return 0.5f * val * (1.0f + std::tanh(tanh_input));
    });

    if (x->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<GeluBackward>(x);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr sigmoid(const TensorPtr& x) {
    auto y = elementwise_unary_op(x, [](float val) {
        return 1.0f / (1.0f + std::exp(-val));
    });
    if (x->requires_grad()) {
        y->set_requires_grad(true);
        auto backward_fn = std::make_shared<SigmoidBackward>(y);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(y, {x}, backward_fn);
        #else
        y->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) accumulate_gradient(x, grads[0]);
            return grads;
        });
        #endif
    }
    return y;
}

TensorPtr tanh_op(const TensorPtr& x) {
    auto y = elementwise_unary_op(x, [](float val) { return std::tanh(val); });
    if (x->requires_grad()) {
        y->set_requires_grad(true);
        auto backward_fn = std::make_shared<TanhBackward>(y);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(y, {x}, backward_fn);
        #else
        y->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) accumulate_gradient(x, grads[0]);
            return grads;
        });
        #endif
    }
    return y;
}

TensorPtr softmax(const TensorPtr& x, int dim) {

    if (dim != -1 && dim != x->ndim() - 1) {
        throw TensorError("softmax only supports last dimension currently");
    }

    const auto& shape = x->shape();
    auto result = zeros(shape, x->dtype(), x->device());

    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }
    int64_t feature_size = shape.back();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_data = x_data + b * feature_size;
        float* batch_result = result_data + b * feature_size;

        float max_val = *std::max_element(batch_data, batch_data + feature_size);

        float sum = 0.0f;
        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] = std::exp(batch_data[i] - max_val);
            sum += batch_result[i];
        }

        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] /= sum;
        }
    }

    if (x->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<SoftmaxBackward>(result, dim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);

            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }

            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr log_softmax(const TensorPtr& x, int dim) {

    if (dim != -1 && dim != x->ndim() - 1) {
        throw TensorError("log_softmax only supports last dimension currently");
    }

    const auto& shape = x->shape();
    auto result = zeros(shape, x->dtype(), x->device());

    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }
    int64_t feature_size = shape.back();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_data = x_data + b * feature_size;
        float* batch_result = result_data + b * feature_size;

        float max_val = *std::max_element(batch_data, batch_data + feature_size);

        float log_sum_exp = 0.0f;
        for (int64_t i = 0; i < feature_size; ++i) {
            log_sum_exp += std::exp(batch_data[i] - max_val);
        }
        log_sum_exp = max_val + std::log(log_sum_exp);

        for (int64_t i = 0; i < feature_size; ++i) {
            batch_result[i] = batch_data[i] - log_sum_exp;
        }
    }

    if (x->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<LogSoftmaxBackward>(x, result, dim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (x->requires_grad()) {
                accumulate_gradient(x, grads[0]);
            }
            return grads;
        });
        #endif
    }

    return result;
}

TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction) {
    if (!shapes_equal(input, target)) {
        throw TensorError("mse_loss: input and target must have the same shape");
    }

    auto diff = sub(input, target);
    auto squared = mul(diff, diff);

    if (reduction == "none") {
        return squared;
    } else if (reduction == "mean") {
        return mean(squared);
    } else if (reduction == "sum") {
        return sum(squared);
    } else {
        throw TensorError("mse_loss: invalid reduction '" + reduction + "'");
    }
}

TensorPtr nll_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction) {
    // input: [N, C] is log-probabilities; target: [N] class indices
    const auto& ishape = input->shape();
    const auto& tshape = target->shape();
    if (ishape.size() != 2) throw TensorError("nll_loss: input must be 2D [N,C]");
    if (tshape.size() != 1) throw TensorError("nll_loss: target must be 1D [N]");
    if (ishape[0] != tshape[0]) throw TensorError("nll_loss: batch size mismatch");
    int64_t N = ishape[0];
    int64_t C = ishape[1];
    std::vector<float> losses(N, 0.0f);
    const float* x = input->data<float>();
    if (target->dtype() == DType::kInt32) {
        const int32_t* t = target->data<int32_t>();
        for (int64_t n = 0; n < N; ++n) {
            int cls = t[n];
            if (cls < 0 || cls >= C) throw TensorError("nll_loss: class index out of range");
            losses[n] = -x[n * C + cls];
        }
    } else {
        const float* t = target->data<float>();
        for (int64_t n = 0; n < N; ++n) {
            int cls = static_cast<int>(t[n]);
            if (cls < 0 || cls >= C) throw TensorError("nll_loss: class index out of range");
            losses[n] = -x[n * C + cls];
        }
    }
    TensorPtr result;
    if (reduction == "none") {
        result = tensor(losses);
    } else if (reduction == "mean") {
        float s = std::accumulate(losses.begin(), losses.end(), 0.0f);
        result = full({1}, s / static_cast<float>(N));
    } else if (reduction == "sum") {
        float s = std::accumulate(losses.begin(), losses.end(), 0.0f);
        result = full({1}, s);
    } else {
        throw TensorError("nll_loss: invalid reduction");
    }
    if (input->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<NLLLossBackward>(input, target, reduction);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {input}, backward_fn);
        #else
        result->set_grad_fn([input, target, reduction, N, C](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            float* gi = grad_input->data<float>();
            const float* targ = target->data<float>();
            float scale = (reduction == "mean") ? (1.0f / static_cast<float>(N)) : 1.0f;
            float g = grad_output->data<float>()[0];
            for (int64_t n = 0; n < N; ++n) {
                int cls = (target->dtype() == DType::kInt32) ? static_cast<int>(target->data<int32_t>()[n]) : static_cast<int>(targ[n]);
                if (cls >= 0 && cls < C) gi[n * C + cls] = -g * scale;
            }
            accumulate_gradient(input, grad_input);
            return {};
        });
        #endif
    }
    return result;
}

TensorPtr batch_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
                     const TensorPtr& running_mean, const TensorPtr& running_var,
                     bool training, float momentum, float eps) {
    // Shape: [N, C, *]
    const auto& ishape = input->shape();
    if (ishape.size() < 2) throw TensorError("batch_norm: input must be at least 2D [N,C,…]");
    int64_t N = ishape[0];
    int64_t C = ishape[1];
    int64_t S = 1;
    for (size_t i = 2; i < ishape.size(); ++i) S *= ishape[i];
    int64_t inner = S;

    auto output = zeros(ishape, input->dtype(), input->device());
    auto mean = zeros({C}, input->dtype(), input->device());
    auto var = zeros({C}, input->dtype(), input->device());
    float* mean_d = mean->data<float>();
    float* var_d = var->data<float>();
    const float* x = input->data<float>();
    float* y = output->data<float>();

    // Compute mean/var (training) or use running stats
    if (training || !running_mean || !running_var) {
        for (int64_t c = 0; c < C; ++c) {
            double m = 0.0;
            double v = 0.0;
            int64_t count = N * inner;
            for (int64_t n = 0; n < N; ++n) {
                const float* x_ptr = x + (n * C + c) * inner;
                for (int64_t s = 0; s < inner; ++s) m += x_ptr[s];
            }
            m /= static_cast<double>(count);
            for (int64_t n = 0; n < N; ++n) {
                const float* x_ptr = x + (n * C + c) * inner;
                for (int64_t s = 0; s < inner; ++s) {
                    double d = static_cast<double>(x_ptr[s]) - m;
                    v += d * d;
                }
            }
            v /= static_cast<double>(count);
            mean_d[c] = static_cast<float>(m);
            var_d[c] = static_cast<float>(v);
        }
        if (running_mean && running_var) {
            float* rm = const_cast<Tensor*>(running_mean.get())->data<float>();
            float* rv = const_cast<Tensor*>(running_var.get())->data<float>();
            for (int64_t c = 0; c < C; ++c) {
                rm[c] = (1.0f - momentum) * rm[c] + momentum * mean_d[c];
                rv[c] = (1.0f - momentum) * rv[c] + momentum * var_d[c];
            }
        }
    } else {
        std::memcpy(mean_d, running_mean->data<float>(), sizeof(float) * C);
        std::memcpy(var_d, running_var->data<float>(), sizeof(float) * C);
    }

    const float* w = weight ? weight->data<float>() : nullptr;
    const float* b = bias ? bias->data<float>() : nullptr;
    for (int64_t c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(var_d[c] + eps);
        for (int64_t n = 0; n < N; ++n) {
            const float* x_ptr = x + (n * C + c) * inner;
            float* y_ptr = y + (n * C + c) * inner;
            for (int64_t s = 0; s < inner; ++s) {
                float xhat = (x_ptr[s] - mean_d[c]) * inv_std;
                float val = xhat;
                if (w) val *= w[c];
                if (b) val += b[c];
                y_ptr[s] = val;
            }
        }
    }

    if (input->requires_grad() || (weight && weight->requires_grad()) || (bias && bias->requires_grad())) {
        output->set_requires_grad(true);
        auto backward_fn = std::make_shared<BatchNormBackward>(input, weight ? weight : nullptr, mean, var, eps);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(output, {input, weight, bias}, backward_fn);
        #else
        output->set_grad_fn([input, weight, bias, mean, var, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            BatchNormBackward impl(input, weight ? weight : nullptr, mean, var, eps);
            auto grads = impl.apply(grad_output);
            if (input->requires_grad()) accumulate_gradient(input, grads[0]);
            if (weight && weight->requires_grad()) accumulate_gradient(weight, grads[1]);
            if (bias && bias->requires_grad()) accumulate_gradient(bias, grads[2]);
            return grads;
        });
        #endif
    }

    return output;
}
TensorPtr layer_norm(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias, float eps) {
    const auto& input_shape = input->shape();
    int64_t normalized_dim = input_shape.back();

    auto result = zeros(input_shape, input->dtype(), input->device());
    const float* input_data = input->data<float>();
    const float* weight_data = weight->data<float>();
    const float* bias_data = bias->data<float>();
    float* result_data = result->data<float>();

    int64_t batch_size = input->numel() / normalized_dim;

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input_data + b * normalized_dim;
        float* batch_result = result_data + b * normalized_dim;

        float mean = 0.0f;
        for (int64_t i = 0; i < normalized_dim; ++i) {
            mean += batch_input[i];
        }
        mean /= normalized_dim;

        float variance = 0.0f;
        for (int64_t i = 0; i < normalized_dim; ++i) {
            float diff = batch_input[i] - mean;
            variance += diff * diff;
        }
        variance /= normalized_dim;

        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int64_t i = 0; i < normalized_dim; ++i) {
            float normalized = (batch_input[i] - mean) * inv_std;
            batch_result[i] = normalized * weight_data[i] + bias_data[i];
        }
    }

    if (input->requires_grad() || weight->requires_grad() || bias->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<LayerNormBackward>(input, weight, bias, nullptr, nullptr, eps);
        register_backward(result, {input, weight, bias}, backward_fn);
        #else
        result->set_grad_fn([input, weight, bias, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            LayerNormBackward impl(input, weight, bias, nullptr, nullptr, eps);
            auto grads = impl.apply(grad_output);
            if (input->requires_grad()) accumulate_gradient(input, grads[0]);
            if (weight->requires_grad()) accumulate_gradient(weight, grads[1]);
            if (bias->requires_grad()) accumulate_gradient(bias, grads[2]);
            return grads;
        });
        #endif
    }

      return result;
  }
  
  TensorPtr silu(const TensorPtr& x) {
      // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      auto result = zeros(x->shape(), x->dtype(), x->device());
      const float* x_data = x->data<float>();
      float* result_data = result->data<float>();
      
      for (int64_t i = 0; i < x->numel(); ++i) {
          float val = x_data[i];
          float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
          result_data[i] = val * sigmoid_val;
      }
      
      if (x->requires_grad()) {
          result->set_requires_grad(true);
          auto backward_fn = std::make_shared<SiLUBackward>(x);
          #ifdef USE_NEW_AUTOGRAD_ENGINE
          register_backward(result, {x}, backward_fn);
          #else
          result->set_grad_fn([backward_fn, x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
              auto grads = backward_fn->apply(grad_output);
              if (x->requires_grad()) accumulate_gradient(x, grads[0]);
              return grads;
          });
          #endif
      }
      
      return result;
  }
  
  TensorPtr rms_norm(const TensorPtr& input, const TensorPtr& weight, float eps) {
      // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
      const auto& input_shape = input->shape();
      int64_t normalized_dim = input_shape.back();

      auto result = zeros(input_shape, input->dtype(), input->device());
      const float* input_data = input->data<float>();
      const float* weight_data = weight->data<float>();
      float* result_data = result->data<float>();

      int64_t batch_size = input->numel() / normalized_dim;

      for (int64_t b = 0; b < batch_size; ++b) {
          const float* batch_input = input_data + b * normalized_dim;
          float* batch_result = result_data + b * normalized_dim;

          // High-precision accumulation to reduce drift in deep stacks
          double square_sum = 0.0;
          for (int64_t i = 0; i < normalized_dim; ++i) {
              square_sum += static_cast<double>(batch_input[i]) * static_cast<double>(batch_input[i]);
          }
          float rms = static_cast<float>(std::sqrt(square_sum / static_cast<double>(normalized_dim) + static_cast<double>(eps)));

          // applyRMSNorm
        for (int64_t i = 0; i < normalized_dim; ++i) {
            // Gemma3: weights are stored as delta; forward multiplies by (1 + weight)
            float scale = 1.0f + weight_data[i];
            batch_result[i] = (batch_input[i] / rms) * scale;
        }
      }

  // [Translated comment removed - see documentation]
  result->set_requires_grad(true);
  #ifdef USE_NEW_AUTOGRAD_ENGINE
  auto backward_fn = std::make_shared<RMSNormBackward>(input, weight, eps);
  register_backward(result, {input, weight}, backward_fn);
  #else
  result->set_grad_fn([input, weight, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
      // [Translated comment removed - see documentation]
      RMSNormBackward impl(input, weight, eps);
      auto grads = impl.apply(grad_output);
      if (input->requires_grad()) accumulate_gradient(input, grads[0]);
      if (weight->requires_grad() && grads.size() > 1) accumulate_gradient(weight, grads[1]);
      return grads;
  });
  #endif

      return result;
  }

  // Variant: RMSNorm with affine scale only (no +1). y = x / rms * weight
  TensorPtr rms_norm_affine(const TensorPtr& input, const TensorPtr& weight, float eps) {
      const auto& input_shape = input->shape();
      int64_t D = input_shape.back();
      auto result = zeros(input_shape, input->dtype(), input->device());
      const float* x = input->data<float>();
      const float* w = weight->data<float>();
      float* y = result->data<float>();
      int64_t batch = input->numel() / D;
      for (int64_t b = 0; b < batch; ++b) {
          const float* xb = x + b * D;
          float* yb = y + b * D;
          double sqsum = 0.0;
          for (int64_t i = 0; i < D; ++i) sqsum += static_cast<double>(xb[i]) * static_cast<double>(xb[i]);
          float inv_rms = static_cast<float>(1.0 / std::sqrt(sqsum / static_cast<double>(D) + static_cast<double>(eps)));
          for (int64_t i = 0; i < D; ++i) {
              yb[i] = xb[i] * inv_rms * w[i];
          }
      }
      if (input->requires_grad() || weight->requires_grad()) {
          result->set_requires_grad(true);
          #ifdef USE_NEW_AUTOGRAD_ENGINE
          auto backward_fn = std::make_shared<RMSNormAffineBackward>(input, weight, eps);
          register_backward(result, {input, weight}, backward_fn);
          #else
          result->set_grad_fn([input, weight, eps](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
              RMSNormAffineBackward impl(input, weight, eps);
              auto grads = impl.apply(grad_output);
              if (input->requires_grad()) accumulate_gradient(input, grads[0]);
              if (weight->requires_grad() && grads.size() > 1) accumulate_gradient(weight, grads[1]);
              return grads;
          });
          #endif
      }
      return result;
  }
  
  TensorPtr cross_entropy_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction) {

    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();

    if (input_shape.size() != 2) {
        throw TensorError("cross_entropy_loss: input must be 2D [batch_size, num_classes]");
    }
    if (target_shape.size() != 1) {
        throw TensorError("cross_entropy_loss: target must be 1D [batch_size]");
    }
    if (input_shape[0] != target_shape[0]) {
        throw TensorError("cross_entropy_loss: batch size mismatch");
    }

    int64_t batch_size = input_shape[0];
    int64_t num_classes = input_shape[1];

    auto log_probs = log_softmax(input, -1);
    const float* log_probs_data = log_probs->data<float>();
    
        // [Translated]
    std::vector<float> losses(batch_size);
    
    // 🔧 Support ignore_index=-100 (to ignore pad tokens)
    const int ignore_index = -100;
    int valid_count = 0;
    
    if (target->dtype() == DType::kInt32) {
        const int32_t* target_data = target->data<int32_t>();
        for (int64_t b = 0; b < batch_size; ++b) {
            int target_class = target_data[b];
            
            // Skip ignore_index (pad token)
            if (target_class == ignore_index) {
                losses[b] = 0.0f;
                continue;
            }
            
            if (target_class < 0 || target_class >= num_classes) {
                throw TensorError("cross_entropy_loss: target class index out of range");
            }
            losses[b] = -log_probs_data[b * num_classes + target_class];
            valid_count++;
        }
    } else if (target->dtype() == DType::kFloat32) {
        const float* target_data = target->data<float>();
        for (int64_t b = 0; b < batch_size; ++b) {
            int target_class = static_cast<int>(target_data[b]);
            
            // Skip ignore_index (pad token)
            if (target_class == ignore_index) {
                losses[b] = 0.0f;
                continue;
            }
            
            if (target_class < 0 || target_class >= num_classes) {
                throw TensorError("cross_entropy_loss: target class index out of range");
            }
            losses[b] = -log_probs_data[b * num_classes + target_class];
            valid_count++;
        }
    } else {
        throw TensorError("cross_entropy_loss: target must be int32 or float32");
    }

    TensorPtr result;
    if (reduction == "none") {
        result = tensor(losses);
    } else if (reduction == "mean") {
        float sum = std::accumulate(losses.begin(), losses.end(), 0.0f);
        // 🔧 Key: for mean reduce only divide by valid samples, excluding ignored pads
        float divisor = (valid_count > 0) ? static_cast<float>(valid_count) : 1.0f;
        result = full({1}, sum / divisor);
    } else if (reduction == "sum") {
        float sum = std::accumulate(losses.begin(), losses.end(), 0.0f);
        result = full({1}, sum);
    } else {
        throw TensorError("cross_entropy_loss: invalid reduction '" + reduction + "'");
    }

    if (input->requires_grad()) {
        result->set_requires_grad(true);
        
        auto backward_fn = std::make_shared<CrossEntropyLossBackward>(input, target, reduction);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {input}, backward_fn);
        #else
        // Legacy recursive backward
        result->set_grad_fn([input, target, reduction, batch_size, num_classes](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            
                        // [Translated]
            auto probs = softmax(input, -1);
            const float* probs_data = probs->data<float>();
            float* grad_data = grad_input->data<float>();
            const float grad_scale = grad_output->data<float>()[0];
            
            // gradientCompute：grad = (softmax - one_hot) * grad_output
            if (target->dtype() == DType::kInt32) {
                const int32_t* target_data = target->data<int32_t>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    int target_class = target_data[b];
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t idx = b * num_classes + c;
                        float prob = probs_data[idx];
                        float one_hot = (c == target_class) ? 1.0f : 0.0f;
                        grad_data[idx] = (prob - one_hot) * grad_scale;
                        if (reduction == "mean") {
                            grad_data[idx] /= batch_size;
                        }
                    }
                }
            } else {
                const float* target_data = target->data<float>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    int target_class = static_cast<int>(target_data[b]);
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t idx = b * num_classes + c;
                        float prob = probs_data[idx];
                        float one_hot = (c == target_class) ? 1.0f : 0.0f;
                        grad_data[idx] = (prob - one_hot) * grad_scale;
                        if (reduction == "mean") {
                            grad_data[idx] /= batch_size;
                        }
                    }
                }
            }
            
            // accumulategradienttoinput
            accumulate_gradient(input, grad_input);
            return {};
        });
        #endif
    }

    return result;
}

TensorPtr reshape(const TensorPtr& tensor, const std::vector<int64_t>& shape) {
    auto result = tensor->reshape(shape);
    
    // settingsgradientpropagate
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        
        auto backward_fn = std::make_shared<ReshapeBackward>(tensor->shape());
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor, original_shape = tensor->shape()](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // [Translated comment removed - see documentation]
            auto grad_input = grad_output->reshape(original_shape);
            accumulate_gradient(tensor, grad_input);
            return {};
        });
        #endif
    }
    
    return result;
}

TensorPtr view(const TensorPtr& tensor, const std::vector<int64_t>& shape) {
    return tensor->view(shape);
}

TensorPtr flatten(const TensorPtr& tensor, int start_dim, int end_dim) {
    const auto& shape = tensor->shape();
    int ndim = shape.size();

    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim || start_dim > end_dim) {
        throw TensorError("flatten: invalid start_dim or end_dim");
    }

    std::vector<int64_t> new_shape;

    for (int i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape[i]);
    }

    int64_t flattened_size = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);

    for (size_t i = end_dim + 1; i < shape.size(); ++i) {
        new_shape.push_back(shape[i]);
    }

    return reshape(tensor, new_shape);
}

TensorPtr squeeze(const TensorPtr& tensor, int dim) {
    return tensor->squeeze(dim);
}

TensorPtr unsqueeze(const TensorPtr& tensor, int dim) {
    return tensor->unsqueeze(dim);
}

TensorPtr sum(const TensorPtr& tensor, int dim, bool keepdim) {

    const auto& shape = tensor->shape();
    int ndim = shape.size();
    
    if (dim != -1) {
                // [Translated]
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw TensorError("sum: dimension out of range");
        }
        
        // Computeresultshape
        std::vector<int64_t> result_shape;
        for (int i = 0; i < ndim; ++i) {
            if (i != dim) {
                result_shape.push_back(shape[i]);
            } else if (keepdim) {
                result_shape.push_back(1);
            }
        }
        
        if (result_shape.empty()) {
            result_shape.push_back(1);
        }
        
        auto result = zeros(result_shape, tensor->dtype(), tensor->device());
        const float* src_data = tensor->data<float>();
        float* dst_data = result->data<float>();
        
                // [Translated]
        int64_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        
        int64_t inner_size = 1;
        for (int i = dim + 1; i < ndim; ++i) {
            inner_size *= shape[i];
        }
        
        int64_t sum_size = shape[dim];
        
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                float sum_val = 0.0f;
                for (int64_t s = 0; s < sum_size; ++s) {
                    int64_t src_idx = outer * (sum_size * inner_size) + s * inner_size + inner;
                    sum_val += src_data[src_idx];
                }
                int64_t dst_idx = outer * inner_size + inner;
                dst_data[dst_idx] = sum_val;
            }
        }
        
        if (tensor->requires_grad()) {
            result->set_requires_grad(true);
            
            auto backward_fn = std::make_shared<SumBackward>(tensor->shape(), dim, keepdim);
            
            #ifdef USE_NEW_AUTOGRAD_ENGINE
            register_backward(result, {tensor}, backward_fn);
            #else
            result->set_grad_fn([tensor, dim, keepdim, original_shape = shape](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
                // sumgradientpropagate：duplicategrad_outputtooriginalshape
                auto grad_expanded = grad_output;
                
                // ifnonekeepdim，requireadddimension
                if (!keepdim) {
                    auto new_shape = grad_output->shape();
                    new_shape.insert(new_shape.begin() + dim, 1);
                    grad_expanded = grad_expanded->reshape(new_shape);
                }
                
                // createwithoriginaltensorsameshapegradient
                auto grad_input = zeros(original_shape, tensor->dtype(), tensor->device());
                const float* grad_data = grad_expanded->data<float>();
                float* input_grad_data = grad_input->data<float>();
                
                // willgradientcopytoallsummedposition
                int64_t total = tensor->numel();
                int64_t repeat_count = original_shape[dim];
                int64_t block_size = grad_expanded->numel();
                
                for (int64_t i = 0; i < total; ++i) {
                    int64_t grad_idx = i / repeat_count % block_size;
                    input_grad_data[i] = grad_data[grad_idx];
                }
                
                accumulate_gradient(tensor, grad_input);
                return {};
            });
            #endif
        }
        
        return result;
    }

    // dim == -1: sum all elements
    const float* data = tensor->data<float>();
    float sum_val = 0.0f;

    for (int64_t i = 0; i < tensor->numel(); ++i) {
        sum_val += data[i];
    }

    std::vector<int64_t> result_shape = keepdim ? tensor->shape() : std::vector<int64_t>{};
    if (keepdim) {
        std::fill(result_shape.begin(), result_shape.end(), 1);
    }

    auto result = full(result_shape.empty() ? std::vector<int64_t>{1} : result_shape, sum_val);

    if (tensor->requires_grad()) {
        result->set_requires_grad(true);

        auto backward_fn = std::make_shared<SumBackward>(tensor->shape(), dim, keepdim);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {

            auto grad_input = zeros(tensor->shape(), tensor->dtype(), tensor->device());
            float grad_val = grad_output->data<float>()[0];

            float* grad_data = grad_input->data<float>();
            for (int64_t i = 0; i < tensor->numel(); ++i) {
                grad_data[i] = grad_val;
            }

            if (tensor->requires_grad()) {
                accumulate_gradient(tensor, grad_input);
            }

            return {grad_input};
        });
        #endif
    }

    return result;
}

TensorPtr mean(const TensorPtr& tensor, int dim, bool keepdim) {
    auto sum_result = sum(tensor, dim, keepdim);
    float count = 0.0f;
    if (dim == -1) {
        count = static_cast<float>(tensor->numel());
    } else {
        int ndim = tensor->ndim();
        int d = dim < 0 ? dim + ndim : dim;
        if (d < 0 || d >= ndim) {
            throw TensorError("mean: dimension out of range");
        }
        count = static_cast<float>(tensor->shape()[d]);
    }
    return div(sum_result, count);
}

bool same_shape(const TensorPtr& a, const TensorPtr& b) {
    return a->shape() == b->shape();
}

bool broadcastable(const TensorPtr& a, const TensorPtr& b) {
    return can_broadcast(a, b);
}

std::vector<int64_t> broadcast_shape(const TensorPtr& a, const TensorPtr& b) {
    return broadcast_shapes(a, b);
}

std::vector<int64_t> infer_broadcast_shape(const TensorPtr& a, const TensorPtr& b) {

    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();

    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result_shape(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == 1) {
            result_shape[max_dims - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            result_shape[max_dims - 1 - i] = dim_a;
        } else if (dim_a == dim_b) {
            result_shape[max_dims - 1 - i] = dim_a;
        } else {
            throw TensorError("Cannot broadcast tensors");
        }
    }

    return result_shape;
}

TensorPtr create_causal_mask(int seq_len, DType dtype, Device device) {
    auto mask = full({seq_len, seq_len}, 0.0f, dtype, device);
    float* mask_data = mask->data<float>();

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask_data[i * seq_len + j] = -1e10f;  // Changed from -1e9f to -1e10f for better masking
            }
        }
    }

    return mask;
}

TensorPtr apply_mask(const TensorPtr& input, const TensorPtr& mask, float mask_value) {

    auto result = input->clone();
    const float* input_data = input->data<float>();
    const float* mask_data = mask->data<float>();
    float* result_data = result->data<float>();

    if (input->shape() == mask->shape()) {
        for (int64_t i = 0; i < input->numel(); ++i) {
            bool keep = mask_data[i] > 0.5f;
            result_data[i] = keep ? input_data[i] : mask_value;
        }
    } else if (input->ndim() == 3 && mask->ndim() == 2) {
        const auto& input_shape = input->shape();
        int64_t batch = input_shape[0];
        int64_t seq_len = input_shape[1];
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t j = 0; j < seq_len; ++j) {
                    int64_t input_idx = b * seq_len * seq_len + i * seq_len + j;
                    int64_t mask_idx = i * seq_len + j;
                    bool keep = mask_data[mask_idx] > 0.5f;
                    result_data[input_idx] = keep ? input_data[input_idx] : mask_value;
                }
            }
        }
    } else if (input->ndim() == 4 && mask->ndim() == 2) {
        const auto& input_shape = input->shape();
        int64_t batch = input_shape[0];
        int64_t heads = input_shape[1];
        int64_t seq_len = input_shape[2];
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t i = 0; i < seq_len; ++i) {
                    for (int64_t j = 0; j < seq_len; ++j) {
                        int64_t input_idx = b * heads * seq_len * seq_len +
                                           h * seq_len * seq_len +
                                           i * seq_len + j;
                        int64_t mask_idx = i * seq_len + j;
                        bool keep = mask_data[mask_idx] > 0.5f;
                        result_data[input_idx] = keep ? input_data[input_idx] : mask_value;
                    }
                }
            }
        }
    } else if (input->ndim() == 2 && mask->ndim() == 2) {
        for (int64_t i = 0; i < input->numel(); ++i) {
            bool keep = mask_data[i] > 0.5f;
            result_data[i] = keep ? input_data[i] : mask_value;
        }
    } else {
        throw TensorError("apply_mask: unsupported shape combination");
    }

    if (input->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ApplyMaskBackward>(input);
        register_backward(result, {input}, backward_fn);
        #else
        result->set_grad_fn([input](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input->shape(), input->dtype(), input->device());
            const float* grad_out_data = grad_output->data<float>();
            float* grad_in_data = grad_input->data<float>();
            for (int64_t i = 0; i < input->numel(); ++i) {
                grad_in_data[i] = grad_out_data[i];
            }
            if (input->requires_grad()) {
                accumulate_gradient(input, grad_input);
            }
            return {grad_input};
        });
        #endif
    }

    return result;
}

TensorPtr repeat_kv_heads(const TensorPtr& kv, int repeat_factor) {
    // kv shape: [batch, kv_heads, seq_len, head_dim]
    // output shape: [batch, kv_heads * repeat_factor, seq_len, head_dim]
    
    auto shape = kv->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("repeat_kv_heads expects 4D tensor");
    }
    
    int64_t batch = shape[0];
    int64_t kv_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];
    
    auto result = zeros({batch, kv_heads * repeat_factor, seq_len, head_dim}, 
                       kv->dtype(), kv->device());
    
    const float* kv_data = kv->data<float>();
    float* result_data = result->data<float>();
    
        // [Translated]
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t kv_h = 0; kv_h < kv_heads; ++kv_h) {
            for (int64_t rep = 0; rep < repeat_factor; ++rep) {
                int64_t out_head = kv_h * repeat_factor + rep;
                for (int64_t s = 0; s < seq_len; ++s) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t kv_idx = b * kv_heads * seq_len * head_dim + 
                                        kv_h * seq_len * head_dim + 
                                        s * head_dim + d;
                        int64_t out_idx = b * (kv_heads * repeat_factor) * seq_len * head_dim + 
                                         out_head * seq_len * head_dim + 
                                         s * head_dim + d;
                        result_data[out_idx] = kv_data[kv_idx];
                    }
                }
            }
        }
    }
    
    // 🔧 Critical fix: force requires_grad so gradients propagate correctly
    // In GQA, repeated K/V must support backpropagation
    result->set_requires_grad(true);
    #ifdef USE_NEW_AUTOGRAD_ENGINE
    auto backward_fn = std::make_shared<RepeatKVHeadsBackward>(repeat_factor);
    register_backward(result, {kv}, backward_fn);
    #else
    if (kv->requires_grad()) {
        result->set_grad_fn([kv, repeat_factor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Legacy: sum repeats back
            const auto& gshape = grad_output->shape();
            int64_t batch = gshape[0], heads_rep=gshape[1], seq=gshape[2], dim=gshape[3];
            int64_t kv_heads = heads_rep / repeat_factor;
            auto grad_kv = zeros({batch, kv_heads, seq, dim}, grad_output->dtype(), grad_output->device());
            const float* src = grad_output->data<float>();
            float* dst = grad_kv->data<float>();
            for (int64_t b=0;b<batch;++b){
                for(int64_t kvh=0;kvh<kv_heads;++kvh){
                    for(int64_t rep=0;rep<repeat_factor;++rep){
                        int64_t out_h = kvh*repeat_factor+rep;
                        for(int64_t s=0;s<seq;++s){
                            for(int64_t d=0;d<dim;++d){
                                int64_t si=(((b*heads_rep+out_h)*seq+s)*dim+d);
                                int64_t di=(((b*kv_heads+kvh)*seq+s)*dim+d);
                                dst[di]+=src[si];
                            }
                        }
                    }
                }
            }
            accumulate_gradient(kv, grad_kv);
            return {grad_kv};
        });
    }
    #endif
    
    return result;
}

TensorPtr apply_rope(const TensorPtr& x, int seq_len, int head_dim, float rope_theta) {
    // x shape: [batch, heads, seq_len, head_dim]
    // RoPE (Rotary Position Embedding) implementsation
    
    auto shape = x->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("apply_rope expects 4D tensor: [batch, heads, seq_len, head_dim]");
    }
    
    int64_t batch = shape[0];
    int64_t heads = shape[1];
    int64_t actual_seq_len = shape[2];
    int64_t actual_head_dim = shape[3];
    
    if (actual_head_dim != head_dim || actual_seq_len != seq_len) {
        throw std::runtime_error("RoPE dimension mismatch");
    }
    
    auto result = zeros(shape, x->dtype(), x->device());
    
    const float* x_data = x->data<float>();
    float* result_data = result->data<float>();
    
        // [Translated]
    std::memcpy(result_data, x_data, batch * heads * seq_len * head_dim * sizeof(float));
    
        // [Translated]
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t pos = 0; pos < seq_len; ++pos) {
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    // Calculate frequency
                    float freq = 1.0f / std::pow(rope_theta, 2.0f * d / head_dim);
                    float angle = pos * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);
                    
                    // Get input indices
                    int64_t idx_base = b * heads * seq_len * head_dim + 
                                      h * seq_len * head_dim + 
                                      pos * head_dim;
                    int64_t idx1 = idx_base + 2 * d;
                    int64_t idx2 = idx_base + 2 * d + 1;
                    
                    // boundarycheck
                    if (idx1 < batch * heads * seq_len * head_dim && 
                        idx2 < batch * heads * seq_len * head_dim) {
                        // Apply rotation
                        float x1 = x_data[idx1];
                        float x2 = x_data[idx2];
                        
                        result_data[idx1] = x1 * cos_val - x2 * sin_val;
                        result_data[idx2] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }
    }
    
    if (x->requires_grad()) {
        result->set_requires_grad(true);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<ApplyRoPEBackward>(seq_len, head_dim, rope_theta);
        register_backward(result, {x}, backward_fn);
        #else
        result->set_grad_fn([x](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // Legacy: pass-through
            accumulate_gradient(x, grad_output);
            return {grad_output};
        });
        #endif
    }
    
    return result;
}

TensorPtr swiglu(const TensorPtr& gate, const TensorPtr& up) {
    // SwiGLU = SiLU(gate) * up
    if (!same_shape(gate, up)) {
        throw std::runtime_error("gate and up tensors must have the same shape for SwiGLU");
    }
    
    auto result = zeros(gate->shape(), gate->dtype(), gate->device());
    const float* gate_data = gate->data<float>();
    const float* up_data = up->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < gate->numel(); ++i) {
        float gate_val = gate_data[i];
        float up_val = up_data[i];
        
        // SiLU(gate) = gate / (1 + exp(-gate))
        float silu_gate = gate_val / (1.0f + std::exp(-gate_val));
        result_data[i] = silu_gate * up_val;
    }
    
    if (gate->requires_grad() || up->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<SwiGLUBackward>(gate, up);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {gate, up}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, gate, up](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (gate->requires_grad()) accumulate_gradient(gate, grads[0]);
            if (up->requires_grad()) accumulate_gradient(up, grads[1]);
            return grads;
        });
        #endif
    }
    
    return result;
}

// LoRA Linearimplements
TensorPtr lora_linear(const TensorPtr& input, const TensorPtr& weight,
                     const TensorPtr& lora_A, const TensorPtr& lora_B,
                     float alpha, const TensorPtr& bias) {
    // Safety checks
    if (!input || !weight || !lora_A || !lora_B) {
        throw TensorError("lora_linear: input, weight, lora_A, lora_B must not be null");
    }
    
    // 🔧 Critical fix: weight is frozen; temporarily stash requires_grad state
    bool weight_requires_grad = weight->requires_grad();
    const_cast<Tensor*>(weight.get())->set_requires_grad(false);
    
    // Main branch: input @ weight (weight frozen, no MatmulBackward registered)
    // 🔧 Auto-correct shape: if weight is [Out, In], transpose to [In, Out]
    TensorPtr main_output;
    {
        const auto& in_shape = input->shape();
        const auto& w_shape = weight->shape();
        int64_t input_in_dim = in_shape.back();
        bool aligned = (w_shape.size() >= 2) && (w_shape[0] == input_in_dim);
        if (!aligned && w_shape.size() >= 2 && w_shape[1] == input_in_dim) {
            // Weight is [Out, In] → transpose to [In, Out]
            auto weight_t = transpose(weight, 0, 1);
            main_output = matmul(input, weight_t);
        } else {
            main_output = matmul(input, weight);
        }
    }
    
    // 🔧 Restore weight requires_grad state
    const_cast<Tensor*>(weight.get())->set_requires_grad(weight_requires_grad);
    
    if (bias) {
        main_output = add(main_output, bias);
    }
    
    // LoRA branch: input @ lora_A @ lora_B * alpha
    // 🔧 Normalize LoRA A/B shapes: A expects [In, Rank], B expects [Rank, Out]
    TensorPtr lora_A_eff = lora_A;
    TensorPtr lora_B_eff = lora_B;
    {
        const auto& in_shape = input->shape();
        int64_t input_in_dim = in_shape.back();
        const auto& a_shape = lora_A->shape();
        const auto& b_shape = lora_B->shape();
        // A: if [Rank, In], transpose to [In, Rank]
        if (a_shape.size() == 2 && a_shape[0] != input_in_dim && a_shape[1] == input_in_dim) {
            lora_A_eff = transpose(lora_A, 0, 1);
        }
        // B: if [Out, Rank], transpose to [Rank, Out]
        if (b_shape.size() == 2) {
            int64_t rank_dim = lora_A_eff->shape()[1];
            if (b_shape[0] != rank_dim && b_shape[1] == rank_dim) {
                lora_B_eff = transpose(lora_B, 0, 1);
            }
        }
    }
    auto lora_hidden = matmul(input, lora_A_eff);
    auto lora_output = matmul(lora_hidden, lora_B_eff);
    auto scaled_lora = mul(lora_output, alpha);
    
    // Merge main and LoRA branches
    auto result = add(main_output, scaled_lora);
    
        // [Translated]
    if (input->requires_grad() || lora_A->requires_grad() || lora_B->requires_grad()) {
        result->set_requires_grad(true);
        
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        // New engine: rely on matmul/add autograd; matmul already fixes batch-dim gradients
        #else
        // Old engine: Use grad_fn (recursive backward)
        result->set_grad_fn([input, weight, lora_A, lora_B, alpha, bias](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            
            // Compute lora_A gradient and accumulate
            if (lora_A->requires_grad()) {
                // 🔧 FIX: use 2D reshape to compute gradients and avoid 3D transpose issues
                auto grad_shape = grad_output->shape();
                auto input_shape = input->shape();
                
                // DEBUG: print shape info
                static int debug_grad_A_count = 0;
                if (false && debug_grad_A_count++ < 5) {  // 🔧 DEBUG output disabled
                    std::cout << "    🔍 DEBUG grad_A compute:" << std::endl;
                    std::cout << "      grad_output shape: [";
                    for (size_t i = 0; i < grad_shape.size(); ++i) {
                        std::cout << grad_shape[i];
                        if (i < grad_shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    std::cout << "      input shape: [";
                    for (size_t i = 0; i < input_shape.size(); ++i) {
                        std::cout << input_shape[i];
                        if (i < input_shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    std::cout << "      lora_A: [" << lora_A->shape()[0] << ", " << lora_A->shape()[1] << "]" << std::endl;
                    std::cout << "      lora_B: [" << lora_B->shape()[0] << ", " << lora_B->shape()[1] << "]" << std::endl;
                }
                
                // Flatten batch dimensions: [B, S, D] → [B*S, D]
                int64_t batch_seq = 1;
                for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
                    batch_seq *= grad_shape[i];
                }
                int64_t out_dim = grad_shape[grad_shape.size() - 1];
                int64_t in_dim = input_shape[input_shape.size() - 1];
                
                auto grad_output_2d = reshape(grad_output, {batch_seq, out_dim});
                auto input_2d = reshape(input, {batch_seq, in_dim});
                
                // 🔧 DEBUG输出已禁用
                
                // grad_lora_hidden = grad_output @ lora_B^T: [B*S, out] @ [out, rank] = [B*S, rank]
                auto lora_B_t = transpose(lora_B, 0, 1);
                
                // 🔧 DEBUG输出已禁用
                
                auto grad_lora_hidden = matmul(grad_output_2d, lora_B_t);
                
                // grad_lora_A = input^T @ grad_lora_hidden: [in, B*S] @ [B*S, rank] = [in, rank]
                auto input_2d_t = transpose(input_2d, 0, 1);
                
                // 🔧 DEBUG输出已禁用
                
                auto grad_lora_A_raw = matmul(input_2d_t, grad_lora_hidden);
                auto grad_lora_A = mul(grad_lora_A_raw, alpha);
                accumulate_gradient(lora_A, grad_lora_A);
            }
            
            // Compute lora_B gradient and accumulate
            if (lora_B->requires_grad()) {
                // 🔧 FIX: likewise use 2D reshape
                auto grad_shape = grad_output->shape();
                auto input_shape = input->shape();
                
                int64_t batch_seq = 1;
                for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
                    batch_seq *= grad_shape[i];
                }
                int64_t out_dim = grad_shape[grad_shape.size() - 1];
                int64_t in_dim = input_shape[input_shape.size() - 1];
                
                auto input_2d = reshape(input, {batch_seq, in_dim});
                auto grad_output_2d = reshape(grad_output, {batch_seq, out_dim});
                
                // lora_hidden = input @ lora_A: [B*S, in] @ [in, rank] = [B*S, rank]
                auto lora_hidden = matmul(input_2d, lora_A);
                
                // grad_lora_B = lora_hidden^T @ grad_output: [rank, B*S] @ [B*S, out] = [rank, out]
                auto lora_hidden_t = transpose(lora_hidden, 0, 1);
                auto grad_lora_B_raw = matmul(lora_hidden_t, grad_output_2d);
                auto grad_lora_B = mul(grad_lora_B_raw, alpha);
                accumulate_gradient(lora_B, grad_lora_B);
            }
            
            // Propagate gradient to input
            if (input->requires_grad()) {
                auto grad_input_main = matmul(grad_output, transpose(weight, 0, 1));
                auto lora_B_t = transpose(lora_B, 0, 1);
                auto lora_A_t = transpose(lora_A, 0, 1);
                auto grad_input_lora = matmul(matmul(grad_output, lora_B_t), lora_A_t);
                auto grad_input_lora_scaled = mul(grad_input_lora, alpha);
                auto grad_input_total = add(grad_input_main, grad_input_lora_scaled);
                accumulate_gradient(input, grad_input_total);
            }
            
            return {};
        });
        #endif
    }
    
    return result;
}

// compareoperatorimplements
TensorPtr eq(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] == data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr ne(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] != data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr gt(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] > data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr lt(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] < data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr ge(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] >= data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

TensorPtr le(const TensorPtr& a, const TensorPtr& b) {
    auto result = zeros(a->shape(), a->dtype(), a->device());
    const float* data_a = a->data<float>();
    const float* data_b = b->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < a->numel(); ++i) {
        result_data[i] = (data_a[i] <= data_b[i]) ? 1.0f : 0.0f;
    }
    
    return result;
}

// [Translated]
TensorPtr abs(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::abs(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<AbsBackward>(tensor);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr sqrt(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::sqrt(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<SqrtBackward>(tensor);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr exp(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::exp(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<ExpBackward>(result);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr log(const TensorPtr& tensor) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::log(data[i]);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<LogBackward>(tensor);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr pow(const TensorPtr& tensor, float exponent) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::pow(data[i], exponent);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<PowBackward>(tensor, exponent);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    
    return result;
}

TensorPtr clamp(const TensorPtr& tensor, float min_val, float max_val) {
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* data = tensor->data<float>();
    float* result_data = result->data<float>();
    
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        result_data[i] = std::min(std::max(data[i], min_val), max_val);
    }
    
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
    }
    
    return result;
}

// =========================================================================
// Data Type Operations (FP32 <-> FP16)
// =========================================================================
TensorPtr dropout(const TensorPtr& tensor, float p, bool training) {
    if (p < 0.0f || p >= 1.0f) {
        throw TensorError("dropout: p must be in [0,1)");
    }
    if (!training || p == 0.0f) {
        return tensor->clone();
    }
    auto result = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    auto mask = zeros(tensor->shape(), tensor->dtype(), tensor->device());
    const float* x = tensor->data<float>();
    float* y = result->data<float>();
    float* m = mask->data<float>();
    float scale = 1.0f / (1.0f - p);
    // 簡單可重現掩碼：以索引生成偽隨機
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        uint32_t seed = static_cast<uint32_t>(i * 1664525u + 1013904223u);
        float r = (seed & 0xFFFFFF) / static_cast<float>(0x1000000);
        m[i] = (r > p) ? 1.0f : 0.0f;
        y[i] = x[i] * m[i] * scale;
    }
    if (tensor->requires_grad()) {
        result->set_requires_grad(true);
        auto backward_fn = std::make_shared<DropoutBackward>(mask, scale);
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        register_backward(result, {tensor}, backward_fn);
        #else
        result->set_grad_fn([backward_fn, tensor](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (tensor->requires_grad()) accumulate_gradient(tensor, grads[0]);
            return grads;
        });
        #endif
    }
    return result;
}
TensorPtr cast(const TensorPtr& tensor, DType target_dtype) {
    if (!tensor) {
        throw TensorError("cast: input tensor is null");
    }

    const DType src_dtype = tensor->dtype();
    if (src_dtype == target_dtype) {
        // [Translated comment removed - see documentation]
        return tensor->clone();
    }

    auto result = std::make_shared<Tensor>(tensor->shape(), target_dtype, tensor->device());

    if (src_dtype == kFloat32 && target_dtype == kFloat16) {
        const float* src = tensor->data<float>();
        uint16_t* dst = result->data<uint16_t>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            dst[i] = float32_to_fp16(src[i]);
        }
        return result;
    }

    if (src_dtype == kFloat16 && target_dtype == kFloat32) {
        const uint16_t* src = tensor->data<uint16_t>();
        float* dst = result->data<float>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            dst[i] = fp16_to_float32(src[i]);
        }
        return result;
    }

    // [Translated comment removed - see documentation]
    if (DTypeUtils::is_floating_point(src_dtype) && DTypeUtils::is_floating_point(target_dtype)) {
        // [Translated comment removed - see documentation]
        TensorPtr as_fp32 = (src_dtype == kFloat32) ? tensor : cast(tensor, kFloat32);
        if (target_dtype == kFloat32) return as_fp32;
        return cast(as_fp32, target_dtype);
    }

    throw TensorError("cast: unsupported dtype conversion");
}

}
