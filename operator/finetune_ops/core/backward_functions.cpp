#include "backward_functions.h"
#include "ops.h"
#include <cmath>
#include <cstring>
#include <string>

namespace ops {

void accumulate_gradient(const TensorPtr& tensor, const TensorPtr& grad) {
    if (!tensor->requires_grad()) {
        return;
    }

    #ifdef USE_NEW_AUTOGRAD_ENGINE
    // New engine handles accumulation in Engine::accumulate_grad
    // This function is only called in legacy mode
    (void)grad;  // Suppress unused warning in new engine mode
    return;
    #else
    // Legacy accumulation with recursive backward
    if (!tensor->grad()) {
        tensor->set_grad(grad->clone());
    } else {
        auto accumulated = add(tensor->grad(), grad);
        tensor->set_grad(accumulated);
    }

    if (tensor->is_leaf()) {
        return;
    } else {
        tensor->backward(grad);
    }
    #endif
}

TensorPtr sum_to_shape(const TensorPtr& tensor, const std::vector<int64_t>& target_shape) {

    if (tensor->shape() == target_shape) {
        return tensor;
    }

    auto result = tensor;
    while (result->shape() != target_shape) {
        if (result->shape().size() > target_shape.size()) {

            result = sum(result, 0, false);
        } else if (result->shape().size() == target_shape.size()) {

            for (size_t i = 0; i < target_shape.size(); ++i) {
                if (target_shape[i] == 1 && result->shape()[i] > 1) {
                    result = sum(result, static_cast<int>(i), true);
                }
            }
            break;
        } else {

            std::vector<int64_t> new_shape = {1};
            new_shape.insert(new_shape.end(), result->shape().begin(), result->shape().end());
            result = reshape(result, new_shape);
        }
    }

    return result;
}

std::vector<TensorPtr> AddBackward::apply(const TensorPtr& grad_output) {

    auto grad_a = sum_to_shape(grad_output, shape_a_);
    auto grad_b = sum_to_shape(grad_output, shape_b_);

    return {grad_a, grad_b};
}

std::vector<TensorPtr> MulBackward::apply(const TensorPtr& grad_output) {

    auto grad_a = mul(grad_output, b_);
    auto grad_b = mul(grad_output, a_);

    grad_a = sum_to_shape(grad_a, a_->shape());
    grad_b = sum_to_shape(grad_b, b_->shape());

    return {grad_a, grad_b};
}

std::vector<TensorPtr> SubBackward::apply(const TensorPtr& grad_output) {
    // d(a - b)/da = 1, d(a - b)/db = -1
    auto grad_a = sum_to_shape(grad_output, shape_a_);
    auto neg_grad = mul(grad_output, -1.0f);
    auto grad_b = sum_to_shape(neg_grad, shape_b_);

    return {grad_a, grad_b};
}

std::vector<TensorPtr> MatmulBackward::apply(const TensorPtr& grad_output) {

    auto grad_a = matmul(grad_output, transpose(b_, -2, -1));

    TensorPtr grad_b = nullptr;
    if (b_ && b_->requires_grad()) {
        const auto& b_shape = b_->shape();
        const auto& go_shape = grad_output->shape();
        const auto& a_shape = a_->shape();

        if (b_shape.size() == 2 && go_shape.size() >= 3) {
            // Batch matmul gradient for shared weights: flatten to [B*, M, K] and [B*, M, N]
            int64_t m = a_shape[a_shape.size() - 2];
            int64_t k = a_shape[a_shape.size() - 1];
            int64_t n = b_shape[1];
            int64_t batch = 1;
            for (size_t i = 0; i + 2 < go_shape.size(); ++i) batch *= go_shape[i];
            auto a2d = reshape(a_, {batch * m, k});          // [B*M, K]
            auto go2d = reshape(grad_output, {batch * m, n}); // [B*M, N]

            const int64_t weight_rows = b_shape[0];
            const int64_t weight_cols = b_shape[1];
            bool weight_in_out = (weight_rows == k && weight_cols == n);   // W shape [in, out]
            bool weight_out_in = (weight_rows == n && weight_cols == k);   // W shape [out, in]

            if (weight_in_out) {
                // grad_W = X^T @ grad_output => [in, out]
                grad_b = matmul(transpose(a2d, 0, 1), go2d);
            } else if (weight_out_in) {
                // grad_W = grad_output^T @ X => [out, in]
                grad_b = matmul(transpose(go2d, 0, 1), a2d);
            } else {
                grad_b = matmul(transpose(a_, -2, -1), grad_output);
            }
        } else {
            grad_b = matmul(transpose(a_, -2, -1), grad_output);
        }

        if (grad_b && grad_b->shape() != b_shape) {
            grad_b = sum_to_shape(grad_b, b_shape);
        }
    }

    return {grad_a, grad_b};
}

std::vector<TensorPtr> ReluBackward::apply(const TensorPtr& grad_output) {

    auto result = zeros(input_->shape(), input_->dtype(), input_->device());

    const float* input_data = input_->data<float>();
    const float* grad_data = grad_output->data<float>();
    float* result_data = result->data<float>();

    for (int64_t i = 0; i < input_->numel(); ++i) {
        result_data[i] = (input_data[i] > 0.0f) ? grad_data[i] : 0.0f;
    }

    return {result};
}

std::vector<TensorPtr> GeluBackward::apply(const TensorPtr& grad_output) {

    auto result = zeros(input_->shape(), input_->dtype(), input_->device());

    const float* input_data = input_->data<float>();
    const float* grad_data = grad_output->data<float>();
    float* result_data = result->data<float>();

    for (int64_t i = 0; i < input_->numel(); ++i) {
        float x = input_data[i];

        float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
        float tanh_val = std::tanh(tanh_input);

        float grad_gelu = 0.5f * (1.0f + tanh_val) +
                         0.5f * x * (1.0f - tanh_val * tanh_val) *
                         0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);

        result_data[i] = grad_data[i] * grad_gelu;
    }

    return {result};
}

std::vector<TensorPtr> SigmoidBackward::apply(const TensorPtr& grad_output) {

    auto one = ones(output_->shape(), output_->dtype(), output_->device());
    auto one_minus_sigmoid = sub(one, output_);
    auto sigmoid_grad = mul(output_, one_minus_sigmoid);
    auto result = mul(grad_output, sigmoid_grad);

    return {result};
}

std::vector<TensorPtr> TanhBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = zeros(output_->shape(), output_->dtype(), output_->device());
    const float* y = output_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    for (int64_t i = 0; i < output_->numel(); ++i) {
        float sech2 = 1.0f - y[i] * y[i];
        gx[i] = gy[i] * sech2;
    }
    return {grad_input};
}

std::vector<TensorPtr> SiLUBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
    const float* x = input_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    for (int64_t i = 0; i < input_->numel(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        float silu_prime = s * (1.0f + x[i] * (1.0f - s));
        gx[i] = gy[i] * silu_prime;
    }
    return {grad_input};
}

std::vector<TensorPtr> DivBackward::apply(const TensorPtr& grad_output) {
    // grad_a = grad / b; grad_b = -grad * a / (b^2)
    auto grad_a = div(grad_output, b_);
    auto b_sq = mul(b_, b_);
    auto neg = mul(mul(grad_output, a_), -1.0f);
    auto grad_b = div(neg, b_sq);
    return {grad_a, grad_b};
}

std::vector<TensorPtr> SqrtBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
    const float* x = input_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    for (int64_t i = 0; i < input_->numel(); ++i) {
        float denom = std::sqrt(x[i]);
        float local = (denom > 0.0f) ? (0.5f / denom) : 0.0f;
        gx[i] = gy[i] * local;
    }
    return {grad_input};
}

std::vector<TensorPtr> ExpBackward::apply(const TensorPtr& grad_output) {
    // dy/dx = y -> grad_input = grad_output * y
    auto result = mul(grad_output, output_);
    return {result};
}

std::vector<TensorPtr> LogBackward::apply(const TensorPtr& grad_output) {
    // dy/dx = 1/x
    auto result = div(grad_output, input_);
    return {result};
}

std::vector<TensorPtr> PowBackward::apply(const TensorPtr& grad_output) {
    // dy/dx = exponent * x^(exponent-1)
    auto x_pow = pow(input_, exponent_ - 1.0f);
    auto scale = mul(x_pow, exponent_);
    auto result = mul(grad_output, scale);
    return {result};
}

std::vector<TensorPtr> AbsBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
    const float* x = input_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    for (int64_t i = 0; i < input_->numel(); ++i) {
        float sign = (x[i] > 0.0f) ? 1.0f : (x[i] < 0.0f ? -1.0f : 0.0f);
        gx[i] = gy[i] * sign;
    }
    return {grad_input};
}

std::vector<TensorPtr> SwiGLUBackward::apply(const TensorPtr& grad_output) {
    // y = SiLU(gate) * up
    auto grad_gate = zeros(gate_->shape(), gate_->dtype(), gate_->device());
    auto grad_up = zeros(up_->shape(), up_->dtype(), up_->device());
    const float* g = gate_->data<float>();
    const float* u = up_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gg = grad_gate->data<float>();
    float* gu = grad_up->data<float>();
    int64_t n = gate_->numel();
    for (int64_t i = 0; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-g[i]));
        float silu = g[i] * s;
        float silu_prime = s * (1.0f + g[i] * (1.0f - s));
        gg[i] = gy[i] * u[i] * silu_prime;
        gu[i] = gy[i] * silu;
    }
    return {grad_gate, grad_up};
}

std::vector<TensorPtr> ScalarOverTensorBackward::apply(const TensorPtr& grad_output) {
    // y = c / x -> dy/dx = -c / x^2
    auto x_sq = mul(input_, input_);
    auto neg_c = mul(full(input_->shape(), scalar_, input_->dtype(), input_->device()), -1.0f);
    auto local = div(neg_c, x_sq);
    auto grad_input = mul(grad_output, local);
    return {grad_input};
}

std::vector<TensorPtr> SoftmaxBackward::apply(const TensorPtr& grad_output) {
    // Implementation specialized for last-dimension softmax (as in ops::softmax)
    // grad_x = y * (grad_output - sum(grad_output * y, dim=-1, keepdim=True))
    const auto& shape = output_->shape();
    const int ndim = static_cast<int>(shape.size());
    int dim = dim_;
    if (dim < 0) dim += ndim;
    // Our softmax forward only supports last dimension; enforce here for correctness
    if (dim != ndim - 1) {
        // Fallback to generic expression if ever needed
        auto softmax_grad_prod = mul(grad_output, output_);
        auto sum_grad = sum(softmax_grad_prod, dim, true);
        auto grad_diff = sub(grad_output, sum_grad);
        auto result = mul(output_, grad_diff);
        return {result};
    }

    int64_t batch_size = 1;
    for (int i = 0; i < ndim - 1; ++i) batch_size *= shape[i];
    int64_t feature_size = shape.back();

    auto grad_input = zeros(shape, grad_output->dtype(), grad_output->device());

    const float* y = output_->data<float>();
    const float* go = grad_output->data<float>();
    float* gx = grad_input->data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* yb = y + b * feature_size;
        const float* gob = go + b * feature_size;
        float* gxb = gx + b * feature_size;
        // s = sum_i(go_i * y_i)
        double s = 0.0;
        for (int64_t i = 0; i < feature_size; ++i) {
            s += static_cast<double>(gob[i]) * static_cast<double>(yb[i]);
        }
        // gx_i = y_i * (go_i - s)
        for (int64_t i = 0; i < feature_size; ++i) {
            gxb[i] = yb[i] * (gob[i] - static_cast<float>(s));
        }
    }

    return {grad_input};
}

std::vector<TensorPtr> LinearBackward::apply(const TensorPtr& grad_output) {

    std::vector<TensorPtr> grads;

    if (!weight_ || !input_ || !grad_output) {
        throw TensorError("LinearBackward: null tensor encountered");
    }

    if (weight_->shape().size() != 2 || input_->shape().size() < 2 || grad_output->shape().size() < 2) {
        throw TensorError("LinearBackward: expected 2D weight and >=2D input/grad_output tensors");
    }

    const auto& weight_shape = weight_->shape();
    int64_t weight_rows = weight_shape[0];
    int64_t weight_cols = weight_shape[1];
    int64_t input_last_dim = input_->shape()[input_->shape().size() - 1];
    int64_t grad_out_last_dim = grad_output->shape()[grad_output->shape().size() - 1];

    TensorPtr grad_input;

    // Prefer weight stored as [in_dim, out_dim] (forward uses matmul(x, W))
    if (weight_cols == grad_out_last_dim && weight_rows == input_last_dim) {
        // Weight stored as [in_dim, out_dim]; grad_input = grad_output @ W
        grad_input = matmul_rhs_T(grad_output, weight_);
    } else if (weight_rows == grad_out_last_dim && weight_cols == input_last_dim) {
        // Weight stored as [out_dim, in_dim]; grad_input = grad_output @ W
        grad_input = matmul(grad_output, weight_);
    } else {
        throw TensorError(
            "LinearBackward: unexpected weight shape [" +
            std::to_string(weight_rows) + ", " + std::to_string(weight_cols) +
            "] for grad_output last dim=" + std::to_string(grad_out_last_dim) +
            ", input last dim=" + std::to_string(input_last_dim));
    }
    grads.push_back(grad_input);

    TensorPtr grad_weight = nullptr;
    if (weight_cols == grad_out_last_dim && weight_rows == input_last_dim) {
        // Weight stored as [in_dim, out_dim]; grad_W = X^T @ grad_output
        grad_weight = matmul(transpose(input_, -2, -1), grad_output);
    } else if (weight_rows == grad_out_last_dim && weight_cols == input_last_dim) {
        // Weight stored as [out_dim, in_dim]; forward equals X @ W^T, grad_W should stay [out_dim, in_dim]
        grad_weight = matmul(transpose(grad_output, -2, -1), input_);
    }
    grads.push_back(grad_weight);

    if (bias_) {
        auto grad_bias = sum(grad_output, 0, false);
        grads.push_back(grad_bias);
    }

    return grads;
}

std::vector<TensorPtr> MSELossBackward::apply(const TensorPtr& grad_output) {

    auto diff = sub(input_, target_);
    auto two = ops::full(diff->shape(), 2.0f, diff->dtype(), diff->device());
    auto grad = mul(two, diff);

    if (reduction_ == "mean") {
        float scale = 1.0f / static_cast<float>(input_->numel());
        auto scale_tensor = ops::full(grad->shape(), scale, grad->dtype(), grad->device());
        grad = mul(grad, scale_tensor);
    }

    grad = mul(grad, grad_output);

    return {grad};
}

std::vector<TensorPtr> TransposeBackward::apply(const TensorPtr& grad_output) {

    auto result = transpose(grad_output, dim0_, dim1_);
    return {result};
}

std::vector<TensorPtr> ReshapeBackward::apply(const TensorPtr& grad_output) {

    auto result = reshape(grad_output, original_shape_);
    return {result};
}

std::vector<TensorPtr> LayerNormBackward::apply(const TensorPtr& grad_output) {
    // Correct LayerNorm backward for last-dimension normalization
    const auto& shape = input_->shape();
    int64_t D = shape.back();
    int64_t batch = input_->numel() / D;

    // Gradients
    auto grad_input = zeros(shape, input_->dtype(), input_->device());
    auto grad_weight = zeros(weight_->shape(), weight_->dtype(), weight_->device());
    auto grad_bias = zeros(weight_->shape(), weight_->dtype(), weight_->device());

    const float* x = input_->data<float>();
    const float* w = weight_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    float* gw = grad_weight->data<float>();
    float* gb = grad_bias->data<float>();

    // Accumulate grad_w and grad_b across batch
    for (int64_t b = 0; b < batch; ++b) {
        const float* xb = x + b * D;
        const float* gyb = gy + b * D;

        // Compute mean and inv_std
        float mean = 0.0f;
        for (int64_t i = 0; i < D; ++i) mean += xb[i];
        mean /= static_cast<float>(D);

        float var = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            float d = xb[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(D);
        float inv_std = 1.0f / std::sqrt(var + eps_);

        // Compute x_hat and helper sums
        float sum_gy = 0.0f;
        float sum_gy_xhat = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            float x_hat = (xb[i] - mean) * inv_std;
            sum_gy += gyb[i] * w[i];
            sum_gy_xhat += (gyb[i] * w[i]) * x_hat;
            gw[i] += gyb[i] * x_hat;
            gb[i] += gyb[i];
        }

        // dx
        for (int64_t i = 0; i < D; ++i) {
            float x_hat = (xb[i] - mean) * inv_std;
            float term = static_cast<float>(D) * (gyb[i] * w[i]) - sum_gy - x_hat * sum_gy_xhat;
            gx[b * D + i] = (inv_std / static_cast<float>(D)) * term;
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

std::vector<TensorPtr> RMSNormBackward::apply(const TensorPtr& grad_output) {
    // Implement numerically stable RMSNorm backward
    const auto& shape = input_->shape();
    int64_t D = shape.back();
    int64_t batch = input_->numel() / D;
    auto grad_input = zeros(shape, input_->dtype(), input_->device());
    auto grad_weight = zeros(weight_->shape(), weight_->dtype(), weight_->device());

    const float* x = input_->data<float>();
    const float* w = weight_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    float* gw = grad_weight->data<float>();

    for (int64_t b=0;b<batch;++b){
        const float* xb = x + b*D;
        const float* gyb = gy + b*D;
        float* gxb = gx + b*D;
        // compute rms and x_hat
        double sqsum = 0.0;
        for (int64_t i=0;i<D;++i) sqsum += static_cast<double>(xb[i]) * static_cast<double>(xb[i]);
        double inv_rms = 1.0 / std::sqrt(sqsum / static_cast<double>(D) + static_cast<double>(eps_));
        // Forward uses w_eff = 1 + w[i]; keep it consistent in backward: y = x_hat * w_eff
        // dL/dx = (gy*w_eff)*(inv_rms) - x * inv_rms^3 * (1/D) * sum_i( (gy*w_eff)*x )
        double dot = 0.0;
        for (int64_t i=0;i<D;++i) {
            double w_eff = 1.0 + static_cast<double>(w[i]);
            dot += static_cast<double>(gyb[i]) * w_eff * static_cast<double>(xb[i]);
        }
        double coeff = (inv_rms*inv_rms*inv_rms) / static_cast<double>(D);
        for (int64_t i=0;i<D;++i){
            double w_eff = 1.0 + static_cast<double>(w[i]);
            double term1 = static_cast<double>(gyb[i]) * w_eff * inv_rms;
            double term2 = static_cast<double>(xb[i]) * coeff * dot;
            gxb[i] = static_cast<float>(term1 - term2);
            gw[i] += static_cast<float>(static_cast<double>(gyb[i]) * static_cast<double>(xb[i]) * inv_rms);  // dL/dw = grad * x_hat
        }
    }

    return {grad_input, grad_weight};
}

std::vector<TensorPtr> RMSNormAffineBackward::apply(const TensorPtr& grad_output) {
    // RMSNorm with affine scale: y = x_hat * w
    const auto& shape = input_->shape();
    int64_t D = shape.back();
    int64_t batch = input_->numel() / D;
    auto grad_input = zeros(shape, input_->dtype(), input_->device());
    auto grad_weight = zeros(weight_->shape(), weight_->dtype(), weight_->device());

    const float* x = input_->data<float>();
    const float* w = weight_->data<float>();
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    float* gw = grad_weight->data<float>();

    for (int64_t b = 0; b < batch; ++b) {
        const float* xb = x + b * D;
        const float* gyb = gy + b * D;
        float* gxb = gx + b * D;
        // inv_rms
        double sqsum = 0.0;
        for (int64_t i = 0; i < D; ++i) sqsum += static_cast<double>(xb[i]) * static_cast<double>(xb[i]);
        double inv_rms = 1.0 / std::sqrt(sqsum / static_cast<double>(D) + static_cast<double>(eps_));
        // dot = sum_i( (gy*w)*x )
        double dot = 0.0;
        for (int64_t i = 0; i < D; ++i) {
            dot += static_cast<double>(gyb[i]) * static_cast<double>(w[i]) * static_cast<double>(xb[i]);
        }
        double coeff = (inv_rms * inv_rms * inv_rms) / static_cast<double>(D);
        for (int64_t i = 0; i < D; ++i) {
            double term1 = static_cast<double>(gyb[i]) * static_cast<double>(w[i]) * inv_rms;
            double term2 = static_cast<double>(xb[i]) * coeff * dot;
            gxb[i] = static_cast<float>(term1 - term2);
            gw[i] += static_cast<float>(static_cast<double>(gyb[i]) * static_cast<double>(xb[i]) * inv_rms);
        }
    }
    return {grad_input, grad_weight};
}

std::vector<TensorPtr> SliceBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = zeros(input_->shape(), input_->dtype(), input_->device());

    const auto& input_shape = input_->shape();
    const auto& out_shape = grad_output->shape();

    std::vector<int64_t> in_strides(input_shape.size());
    in_strides.back() = 1;
    for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
    }

    std::vector<int64_t> out_strides(out_shape.size());
    out_strides.back() = 1;
    for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    const float* gout = grad_output->data<float>();
    float* gin = grad_input->data<float>();

    std::function<void(int,int64_t,int64_t)> scatter = [&](int d, int64_t out_offset, int64_t in_offset) {
        if (d == static_cast<int>(input_shape.size())) {
            gin[in_offset] += gout[out_offset];
            return;
        }
        int64_t steps = (d == dim_) ? length_ : input_shape[d];
        int64_t in_start = (d == dim_) ? start_ : 0;
        for (int64_t i = 0; i < steps; ++i) {
            scatter(d + 1,
                    out_offset + i * out_strides[d],
                    in_offset + (in_start + i) * in_strides[d]);
        }
    };
    scatter(0, 0, 0);

    return {grad_input};
}

std::vector<TensorPtr> CrossEntropyLossBackward::apply(const TensorPtr& grad_output) {
    
    const auto& input_shape = input_->shape();
    
    int64_t batch_size = input_shape[0];
    int64_t num_classes = input_shape[1];
    
    auto softmax_probs = softmax(input_, -1);
    const float* softmax_data = softmax_probs->data<float>();
    const float* target_data = target_->data<float>();
    
    auto grad_input = zeros(input_shape, input_->dtype(), input_->device());
    float* grad_data = grad_input->data<float>();
    
    // 🔧 Support ignore_index=-100: count only valid samples
    const int ignore_index = -100;
    int valid_count = 0;
    for (int64_t b = 0; b < batch_size; ++b) {
        int target_class = static_cast<int>(target_data[b]);
        if (target_class != ignore_index && target_class >= 0 && target_class < num_classes) {
            valid_count++;
        }
    }
    
    float scale = 1.0f;
    if (reduction_ == "mean") {
        // Divide only by the valid sample count
        scale = (valid_count > 0) ? (1.0f / static_cast<float>(valid_count)) : 1.0f;
    }
    
    for (int64_t b = 0; b < batch_size; ++b) {
        int target_class = static_cast<int>(target_data[b]);
        
        // Skip ignore_index (pad token)
        if (target_class == ignore_index) {
            continue;  // Gradient stays 0
        }
        
        if (target_class >= 0 && target_class < num_classes) {
            for (int64_t c = 0; c < num_classes; ++c) {
                int64_t idx = b * num_classes + c;
                if (c == target_class) {
                    grad_data[idx] = (softmax_data[idx] - 1.0f) * scale;
                } else {
                    grad_data[idx] = softmax_data[idx] * scale;
                }
            }
        }
    }
    
    // [Translated comment removed - see documentation]
    float grad_scale = 1.0f;
    if (grad_output->numel() == 1) {
        grad_scale = grad_output->data<float>()[0];
    }
    
    if (grad_scale != 1.0f) {
        grad_input = mul(grad_input, grad_scale);
    }
    
    return {grad_input};
}

std::vector<TensorPtr> LogSoftmaxBackward::apply(const TensorPtr& grad_output) {
    
    auto softmax_probs = softmax(input_, dim_);
    int dim = dim_;
    if (dim < 0) {
        dim += static_cast<int>(grad_output->shape().size());
    }
    auto sum_grad = sum(grad_output, dim, true);
    auto grad_input = sub(grad_output, mul(softmax_probs, sum_grad));
    
    return {grad_input};
}

std::vector<TensorPtr> PermuteBackward::apply(const TensorPtr& grad_output) {
    auto grad_input = permute(grad_output, inv_dims_);
    return {grad_input};
}

std::vector<TensorPtr> RepeatKVHeadsBackward::apply(const TensorPtr& grad_output) {
    const auto& gshape = grad_output->shape();
    if (gshape.size() != 4) {
        throw TensorError("RepeatKVHeadsBackward expects 4D gradient");
    }
    int64_t batch = gshape[0];
    int64_t heads_rep = gshape[1];
    int64_t seq = gshape[2];
    int64_t dim = gshape[3];
    if (heads_rep % repeat_factor_ != 0) {
        throw TensorError("repeat_factor does not divide head dimension");
    }
    int64_t kv_heads = heads_rep / repeat_factor_;
    
    auto grad_kv = zeros({batch, kv_heads, seq, dim}, grad_output->dtype(), grad_output->device());
    const float* src = grad_output->data<float>();
    float* dst = grad_kv->data<float>();
    for (int64_t b=0;b<batch;++b){
        for(int64_t kv=0;kv<kv_heads;++kv){
            for(int64_t rep=0;rep<repeat_factor_;++rep){
                int64_t out_head = kv*repeat_factor_+rep;
                for(int64_t s=0;s<seq;++s){
                    for(int64_t d=0;d<dim;++d){
                        int64_t src_idx = (((b*heads_rep + out_head)*seq + s)*dim + d);
                        int64_t dst_idx = (((b*kv_heads + kv)*seq + s)*dim + d);
                        dst[dst_idx] += src[src_idx];
                    }
                }
            }
        }
    }
    
    return {grad_kv};
}

std::vector<TensorPtr> ApplyRoPEBackward::apply(const TensorPtr& grad_output) {
    const auto& gshape = grad_output->shape();
    if (gshape.size() != 4) {
        throw TensorError("ApplyRoPEBackward expects 4D tensor [batch, heads, seq, head_dim]");
    }
    int64_t batch = gshape[0];
    int64_t heads = gshape[1];
    int64_t seq_len = gshape[2];
    int64_t head_dim = gshape[3];
    if (head_dim != head_dim_) {
        // Allow silent continue but check consistency
    }
    if ((head_dim % 2) != 0) {
        throw TensorError("ApplyRoPEBackward requires even head_dim");
    }

    auto grad_input = zeros(gshape, grad_output->dtype(), grad_output->device());
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();

    const int64_t stride_head = seq_len * head_dim;
    const int64_t stride_batch = heads * stride_head;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t pos = 0; pos < seq_len; ++pos) {
                int64_t base = b * stride_batch + h * stride_head + pos * head_dim;
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    float freq = 1.0f / std::pow(rope_theta_, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
                    float angle = static_cast<float>(pos) * freq;
                    float c = std::cos(angle);
                    float s = std::sin(angle);
                    int64_t idx1 = base + 2 * d;
                    int64_t idx2 = base + 2 * d + 1;
                    float g1p = gy[idx1];
                    float g2p = gy[idx2];
                    // grad_input = R^T * grad_output
                    gx[idx1] = g1p * c + g2p * s;
                    gx[idx2] = -g1p * s + g2p * c;
                }
            }
        }
    }

    return {grad_input};
}

std::vector<TensorPtr> MemoryFirstMLPBackward::apply(const TensorPtr& grad_output) {
    // Recomputation-based backward (consistent with memory_first_mlp.cpp implementsation)
    auto grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
    
    const float* input_data = input_->data<float>();
    const float* fc_w_data = fc_weight_->data<float>();
    const float* proj_w_data = proj_weight_->data<float>();
    const float* grad_out_data = grad_output->data<float>();
    float* grad_in_data = grad_input->data<float>();
    
    // Backward propagation chunk by chunk
    for (int64_t chunk_start = 0; chunk_start < n_inner_; chunk_start += chunk_size_) {
        int64_t chunk_end = std::min(chunk_start + chunk_size_, n_inner_);
        int64_t actual_chunk_size = chunk_end - chunk_start;
        
        // Recompute forward intermediate values
        std::vector<float> hidden_chunk(batch_seq_ * actual_chunk_size);
        std::vector<float> gelu_grad(batch_seq_ * actual_chunk_size);
        
        // Recompute hidden & gelu_grad
        for (int64_t i = 0; i < batch_seq_; ++i) {
            for (int64_t j = 0; j < actual_chunk_size; ++j) {
                int64_t fc_row = chunk_start + j;
                float sum = fc_bias_->data<float>()[fc_row];
                for (int64_t k = 0; k < n_embd_; ++k) {
                    sum += input_data[i * n_embd_ + k] * fc_w_data[fc_row * n_embd_ + k];
                }
                hidden_chunk[i * actual_chunk_size + j] = sum;
                
                // GELU derivative
                float x = sum;
                float tanh_input = 0.7978845608f * (x + 0.044715f * x * x * x);
                float tanh_val = std::tanh(tanh_input);
                float sech2 = 1.0f - tanh_val * tanh_val;
                float tanh_grad = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                gelu_grad[i * actual_chunk_size + j] = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * tanh_grad;
            }
        }
        
        // Backward: grad_activated -> grad_hidden -> grad_input
        std::vector<float> grad_activated_chunk(batch_seq_ * actual_chunk_size, 0.0f);
        for (int64_t i = 0; i < batch_seq_; ++i) {
            for (int64_t k = 0; k < actual_chunk_size; ++k) {
                float sum = 0.0f;
                for (int64_t j = 0; j < n_embd_; ++j) {
                    int64_t proj_col = chunk_start + k;
                    sum += grad_out_data[i * n_embd_ + j] * proj_w_data[j * n_inner_ + proj_col];
                }
                grad_activated_chunk[i * actual_chunk_size + k] = sum * gelu_grad[i * actual_chunk_size + k];
            }
        }
        
        // Propagate to input
        for (int64_t i = 0; i < batch_seq_; ++i) {
            for (int64_t j = 0; j < n_embd_; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < actual_chunk_size; ++k) {
                    int64_t fc_row = chunk_start + k;
                    sum += grad_activated_chunk[i * actual_chunk_size + k] * fc_w_data[fc_row * n_embd_ + j];
                }
                grad_in_data[i * n_embd_ + j] += sum;
            }
        }
    }
    
    return {grad_input};
}

std::vector<TensorPtr> SumBackward::apply(const TensorPtr& grad_output) {
    // Properly broadcast grad_output back to input_shape_ along dim_
    auto grad_input = zeros(input_shape_, grad_output->dtype(), grad_output->device());
    float* gx = grad_input->data<float>();
    const float* go = grad_output->data<float>();

    // Sum over all elements
    if (dim_ == -1) {
        float g = go[0];
        for (int64_t i = 0; i < grad_input->numel(); ++i) gx[i] = g;
        return {grad_input};
    }

    // Sum over a specific dimension
    // Compute outer/inner sizes relative to dim_
    int ndim = static_cast<int>(input_shape_.size());
    int dim = dim_;
    if (dim < 0) dim += ndim;

    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= input_shape_[i];
    int64_t sum_size = input_shape_[dim];
    int64_t inner = 1;
    for (int i = dim + 1; i < ndim; ++i) inner *= input_shape_[i];

    // grad_output shape depends on keepdim_
    // If keepdim_, its shape matches input with dimension dim set to 1
    // Otherwise, grad_output lacks that dimension
    // Indexing helper to read grad_output at (outer_idx, inner_idx)
    auto read_go = [&](int64_t outer_idx, int64_t inner_idx) -> float {
        if (keepdim_) {
            // go layout: [outer, 1, inner]
            int64_t idx = outer_idx * inner + inner_idx;
            return go[idx];
        } else {
            // go layout: [outer, inner]
            int64_t idx = outer_idx * inner + inner_idx;
            return go[idx];
        }
    };

    // Fill grad_input by repeating grad_output across sum dimension
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < sum_size; ++s) {
            for (int64_t i = 0; i < inner; ++i) {
                int64_t dst = (o * sum_size + s) * inner + i;
                gx[dst] = read_go(o, i);
            }
        }
    }

    return {grad_input};
}

std::vector<TensorPtr> BatchNormBackward::apply(const TensorPtr& grad_output) {
    // Shapes: input [N, C, *], weight [C]
    const auto& ishape = input_->shape();
    int64_t N = ishape[0];
    int64_t C = ishape[1];
    int64_t S = 1;
    for (size_t i = 2; i < ishape.size(); ++i) S *= ishape[i];
    int64_t inner = S;

    auto grad_input = zeros(ishape, input_->dtype(), input_->device());
    auto grad_weight = zeros({C}, weight_->dtype(), weight_->device());
    auto grad_bias = zeros({C}, weight_->dtype(), weight_->device());

    const float* x = input_->data<float>();
    const float* w = weight_ ? weight_->data<float>() : nullptr;
    const float* gy = grad_output->data<float>();
    float* gx = grad_input->data<float>();
    float* gw = grad_weight->data<float>();
    float* gb = grad_bias->data<float>();

    // mean/var already computed in forward (passed in)
    const float* mean = mean_->data<float>();
    const float* var = var_->data<float>();

    for (int64_t c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(var[c] + eps_);
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const float* x_ptr = x + (n * C + c) * inner;
            const float* gy_ptr = gy + (n * C + c) * inner;
            for (int64_t s = 0; s < inner; ++s) {
                float xhat = (x_ptr[s] - mean[c]) * inv_std;
                float dy = gy_ptr[s] * (w ? w[c] : 1.0f);
                sum_dy += dy;
                sum_dy_xhat += dy * xhat;
            }
        }
        gw[c] = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const float* x_ptr = x + (n * C + c) * inner;
            const float* gy_ptr = gy + (n * C + c) * inner;
            for (int64_t s = 0; s < inner; ++s) {
                float xhat = (x_ptr[s] - mean[c]) * inv_std;
                gw[c] += gy_ptr[s] * xhat;
                gb[c] += gy_ptr[s];
            }
        }
        for (int64_t n = 0; n < N; ++n) {
            const float* x_ptr = x + (n * C + c) * inner;
            const float* gy_ptr = gy + (n * C + c) * inner;
            float* gx_ptr = gx + (n * C + c) * inner;
            for (int64_t s = 0; s < inner; ++s) {
                float dy_w = gy_ptr[s] * (w ? w[c] : 1.0f);
                float xhat = (x_ptr[s] - mean[c]) * inv_std;
                float term = static_cast<float>(inner * N) * dy_w - sum_dy - xhat * sum_dy_xhat;
                gx_ptr[s] = (inv_std / static_cast<float>(inner * N)) * term;
            }
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

std::vector<TensorPtr> NLLLossBackward::apply(const TensorPtr& grad_output) {
    const auto& ishape = input_->shape(); // [N, C]
    int64_t N = ishape[0];
    int64_t C = ishape[1];
    auto grad_input = zeros(ishape, input_->dtype(), input_->device());
    float* gi = grad_input->data<float>();
    const float* target = target_->data<float>();
    float scale = 1.0f;
    if (reduction_ == "mean") scale = 1.0f / static_cast<float>(N);
    float g = grad_output->data<float>()[0];
    for (int64_t n = 0; n < N; ++n) {
        int t = static_cast<int>(target[n]);
        if (t >= 0 && t < C) {
            gi[n * C + t] = -g * scale;
        }
    }
    return {grad_input};
}

std::vector<TensorPtr> ScaleBackward::apply(const TensorPtr& grad_output) {
    // dy/dx = scale -> grad_input = grad_output * scale
    auto grad_input = mul(grad_output, scale_);
    return {grad_input};
}

std::vector<TensorPtr> PassThroughBackward::apply(const TensorPtr& grad_output) {
    // dy/dx = 1 -> grad_input = grad_output
    return {grad_output};
}

std::vector<TensorPtr> DropoutBackward::apply(const TensorPtr& grad_output) {
    // grad_input = grad_output * mask * scale
    auto scaled = mul(grad_output, scale_);
    auto grad_input = mul(scaled, mask_);
    return {grad_input};
}

std::vector<TensorPtr> ApplyMaskBackward::apply(const TensorPtr& grad_output) {
    // y = input + mask (mask is constant), so grad wrt input is grad_output, mask has no grad
    return {grad_output};
}

std::vector<TensorPtr> MatmulRhsTBackward::apply(const TensorPtr& grad_output) {
    // y = a @ b^T
    // grad_a = grad_y @ b
    // grad_b = grad_y^T @ a  (then transpose back to match original b shape semantics)
    std::vector<TensorPtr> grads;
    TensorPtr grad_a = nullptr;
    TensorPtr grad_b = nullptr;
    if (a_ && a_->requires_grad()) {
        grad_a = matmul(grad_output, b_);
    }
    if (b_ && b_->requires_grad()) {
        auto grad_y_T = transpose(grad_output, -2, -1);
        grad_b = matmul(grad_y_T, a_);
    }
    grads.push_back(grad_a);
    grads.push_back(grad_b);
    return grads;
}

std::vector<TensorPtr> LoRALinearBackward::apply(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads;
    
    // Gradient w.r.t. input (from both main and LoRA branches)
    TensorPtr grad_input;
    if (input_->requires_grad()) {
        // DEBUG disabled to improve performance
        
        // Main branch: grad_output @ weight^T
        // 🔧 Robust handling: external systems may swap weight shape to [output_dim, input_dim]
        // Expected:
        //  - input_.shape()   [..., in_dim]
        //  - grad_output      [..., out_dim]
        //  - weight           [in_dim, out_dim] or [out_dim, in_dim]
        int64_t in_dim  = input_->shape()[input_->shape().size() - 1];
        int64_t out_dim = grad_output->shape()[grad_output->shape().size() - 1];

        auto w0 = weight_->shape()[0];
        auto w1 = weight_->shape()[1];

        if (w0 != in_dim && w1 != out_dim && !(w0 == out_dim && w1 == in_dim)) {
            throw TensorError("LoRA backward: unexpected weight shape [" + std::to_string(w0) + ", " + std::to_string(w1) + "]");
        }

        const auto& input_shape = input_->shape();
        int64_t batch_seq = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
            batch_seq *= input_shape[i];
        }

        const float* grad_out_data = grad_output->data<float>();
        const float* weight_data = weight_->data<float>();
        const float* lora_A_data = lora_A_->data<float>();
        const float* lora_B_data = lora_B_->data<float>();

        auto grad_input_main = zeros(input_shape, grad_output->dtype(), grad_output->device());
        float* grad_input_main_data = grad_input_main->data<float>();

        if (w0 == in_dim && w1 == out_dim) {
            for (int64_t b = 0; b < batch_seq; ++b) {
                for (int64_t i = 0; i < in_dim; ++i) {
                    float sum = 0.0f;
                    for (int64_t o = 0; o < out_dim; ++o) {
                        sum += grad_out_data[b * out_dim + o] * weight_data[i * out_dim + o];
                    }
                    grad_input_main_data[b * in_dim + i] = sum;
                }
            }
        } else { // weight layout [out_dim, in_dim]
            for (int64_t b = 0; b < batch_seq; ++b) {
                for (int64_t i = 0; i < in_dim; ++i) {
                    float sum = 0.0f;
                    for (int64_t o = 0; o < out_dim; ++o) {
                        sum += grad_out_data[b * out_dim + o] * weight_data[o * in_dim + i];
                    }
                    grad_input_main_data[b * in_dim + i] = sum;
                }
            }
        }

        auto grad_hidden = zeros({batch_seq, lora_B_->shape()[0]}, grad_output->dtype(), grad_output->device());
        float* grad_hidden_data = grad_hidden->data<float>();
        int64_t rank = lora_B_->shape()[0];

        for (int64_t b = 0; b < batch_seq; ++b) {
            for (int64_t r = 0; r < rank; ++r) {
                float sum = 0.0f;
                for (int64_t o = 0; o < out_dim; ++o) {
                    sum += grad_out_data[b * out_dim + o] * lora_B_data[r * out_dim + o];
                }
                grad_hidden_data[b * rank + r] = sum;
            }
        }

        auto grad_input_lora_flat = zeros({batch_seq, in_dim}, grad_output->dtype(), grad_output->device());
        float* grad_input_lora_flat_data = grad_input_lora_flat->data<float>();

        for (int64_t b = 0; b < batch_seq; ++b) {
            for (int64_t i = 0; i < in_dim; ++i) {
                float sum = 0.0f;
                for (int64_t r = 0; r < rank; ++r) {
                    sum += grad_hidden_data[b * rank + r] * lora_A_data[i * rank + r];
                }
                grad_input_lora_flat_data[b * in_dim + i] = sum * alpha_;
            }
        }

        auto grad_input_lora = zeros(input_shape, grad_output->dtype(), grad_output->device());
        std::memcpy(grad_input_lora->data<float>(), grad_input_lora_flat_data, sizeof(float) * batch_seq * in_dim);

        grad_input = add(grad_input_main, grad_input_lora);
    } else {
        grad_input = nullptr;
    }
    grads.push_back(grad_input);
    
    // Gradient w.r.t. weight (frozen, returns nullptr)
    grads.push_back(nullptr);
    
    // Gradient w.r.t. lora_A
    TensorPtr grad_lora_A;
    if (lora_A_->requires_grad()) {
        // 🔧 Complete fix: use pure numeric computation, bypass autograd
        auto grad_shape = grad_output->shape();
        auto input_shape = input_->shape();
        
        int64_t batch_seq = 1;
        for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
            batch_seq *= grad_shape[i];
        }
        int64_t out_dim = grad_shape[grad_shape.size() - 1];
        int64_t in_dim = input_shape[input_shape.size() - 1];
        int64_t rank = lora_A_->shape()[1];
        
        // Manually compute grad_lora_A without triggering autograd
        // grad_lora_A = input^T @ (grad_output @ lora_B^T) * alpha
        
        // Step 1: grad_output_2d @ lora_B^T -> grad_lora_hidden
        auto grad_lora_hidden = zeros({batch_seq, rank});
        const float* grad_out = grad_output->data<float>();
        const float* lora_B_data = lora_B_->data<float>();
        float* hidden_data = grad_lora_hidden->data<float>();
        
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t r = 0; r < rank; ++r) {
                float sum = 0.0f;
                for (int64_t o = 0; o < out_dim; ++o) {
                    sum += grad_out[i * out_dim + o] * lora_B_data[r * out_dim + o];
                }
                hidden_data[i * rank + r] = sum;
            }
        }
        
        // Step 2: input^T @ grad_lora_hidden -> grad_lora_A
        grad_lora_A = zeros({in_dim, rank});
        const float* input_data = input_->data<float>();
        float* grad_A_data = grad_lora_A->data<float>();
        
        for (int64_t d = 0; d < in_dim; ++d) {
            for (int64_t r = 0; r < rank; ++r) {
                float sum = 0.0f;
                for (int64_t i = 0; i < batch_seq; ++i) {
                    sum += input_data[i * in_dim + d] * hidden_data[i * rank + r];
                }
                grad_A_data[d * rank + r] = sum * alpha_;
            }
        }
    } else {
        grad_lora_A = nullptr;
    }
    grads.push_back(grad_lora_A);
    
    // Gradient w.r.t. lora_B
    TensorPtr grad_lora_B;
    if (lora_B_->requires_grad()) {
        // 🔧 Complete fix: use pure numeric computation, bypass autograd
        auto grad_shape = grad_output->shape();
        auto input_shape = input_->shape();
        
        int64_t batch_seq = 1;
        for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
            batch_seq *= grad_shape[i];
        }
        int64_t out_dim = grad_shape[grad_shape.size() - 1];
        int64_t in_dim = input_shape[input_shape.size() - 1];
        int64_t rank = lora_A_->shape()[1];
        
        // Manually compute grad_lora_B without triggering autograd
        // grad_lora_B = (input @ lora_A)^T @ grad_output * alpha
        
        // Step 1: input @ lora_A -> lora_hidden
        auto lora_hidden = zeros({batch_seq, rank});
        const float* input_data = input_->data<float>();
        const float* lora_A_data = lora_A_->data<float>();
        float* hidden_data = lora_hidden->data<float>();
        
        for (int64_t i = 0; i < batch_seq; ++i) {
            for (int64_t r = 0; r < rank; ++r) {
                float sum = 0.0f;
                for (int64_t d = 0; d < in_dim; ++d) {
                    sum += input_data[i * in_dim + d] * lora_A_data[d * rank + r];
                }
                hidden_data[i * rank + r] = sum;
            }
        }
        
        // Step 2: lora_hidden^T @ grad_output -> grad_lora_B
        grad_lora_B = zeros({rank, out_dim});
        const float* grad_out = grad_output->data<float>();
        float* grad_B_data = grad_lora_B->data<float>();
        
        for (int64_t r = 0; r < rank; ++r) {
            for (int64_t o = 0; o < out_dim; ++o) {
                float sum = 0.0f;
                for (int64_t i = 0; i < batch_seq; ++i) {
                    sum += hidden_data[i * rank + r] * grad_out[i * out_dim + o];
                }
                grad_B_data[r * out_dim + o] = sum * alpha_;
            }
        }
    } else {
        grad_lora_B = nullptr;
    }
    grads.push_back(grad_lora_B);
    
    // Gradient w.r.t. bias (if exists)
    if (bias_ && bias_->requires_grad()) {
        auto grad_bias = sum(grad_output, 0, false);
        grads.push_back(grad_bias);
    } else {
        grads.push_back(nullptr);
    }
    
    return grads;
}

}
