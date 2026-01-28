#include "optimizer.h"
#include <cmath>

namespace ops {

void Optimizer::clip_grad_norm(const std::vector<TensorPtr>& parameters, float max_norm) {
    float total_norm = compute_grad_norm(parameters);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        scale_gradients(parameters, scale);
    }
}

float Optimizer::compute_grad_norm(const std::vector<TensorPtr>& parameters) {
    float total_norm = 0.0f;
    for (auto& param : parameters) {
        if (param->grad()) {
            auto grad = param->grad();
            const float* grad_data = grad->data<float>();
            for (int64_t i = 0; i < grad->numel(); ++i) {
                total_norm += grad_data[i] * grad_data[i];
            }
        }
    }
    return std::sqrt(total_norm);
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
