#include "optimizer.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ops {

void Optimizer::clip_grad_norm(const std::vector<TensorPtr>& parameters, float max_norm) {
    if (!std::isfinite(max_norm) || max_norm < 0)
        throw std::invalid_argument("max_norm must be finite and nonnegative");
    float total_norm = compute_grad_norm(parameters);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        scale_gradients(parameters, scale);
    }
}

float Optimizer::compute_grad_norm(const std::vector<TensorPtr>& parameters) {
    double total_norm = 0.0;
    for (auto& param : parameters) {
        if (!param)
            throw std::invalid_argument("Null parameter in gradient norm");
        if (param->grad()) {
            auto grad = param->grad();
            if (grad->dtype() != kFloat32 || grad->shape() != param->shape())
                throw std::invalid_argument("Gradient norm requires matching FP32 gradients");
            const float* grad_data = grad->data<float>();
            for (int64_t i = 0; i < grad->numel(); ++i) {
                const double value = grad_data[i];
                if (!std::isfinite(value))
                    throw std::runtime_error("Non-finite gradient norm");
                total_norm += value * value;
            }
        }
    }
    const double norm = std::sqrt(total_norm);
    if (!std::isfinite(norm) || norm > std::numeric_limits<float>::max())
        throw std::runtime_error("Gradient norm exceeds FP32 range");
    return static_cast<float>(norm);
}

void Optimizer::scale_gradients(const std::vector<TensorPtr>& parameters, float scale) {
    for (auto& param : parameters) {
        if (param->grad()) {
            auto grad = param->grad();
            float* grad_data = grad->data<float>();
            for (int64_t i = 0; i < grad->numel(); ++i) {
                grad_data[i] *= scale;
            }
        }
    }
}

} // namespace ops
