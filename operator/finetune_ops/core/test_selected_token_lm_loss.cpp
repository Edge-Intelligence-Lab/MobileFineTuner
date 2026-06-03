#include "lm_loss.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* ad = a->data<float>();
    const float* bd = b->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ad[i] - bd[i]));
    }
    return max_diff;
}

void require_close(float actual, float expected, float tol, const std::string& name) {
    if (std::fabs(actual - expected) > tol) {
        std::cerr << name << " mismatch: actual=" << actual
                  << " expected=" << expected << std::endl;
        std::exit(1);
    }
}

void require_tensor_close(const TensorPtr& actual, const TensorPtr& expected,
                          float tol, const std::string& name) {
    const float diff = max_abs_diff(actual, expected);
    if (diff > tol) {
        std::cerr << name << " mismatch: max_abs_diff=" << diff << std::endl;
        std::exit(1);
    }
}

} // namespace

int main() {
    const int64_t B = 2;
    const int64_t S = 4;
    const int64_t H = 3;
    const int64_t V = 5;

    std::vector<float> hidden_values = {
        0.10f, -0.20f, 0.30f,
        0.00f, 0.25f, -0.15f,
        0.40f, -0.10f, 0.05f,
        -0.30f, 0.20f, 0.10f,
        0.05f, 0.15f, -0.25f,
        0.30f, 0.10f, -0.05f,
        -0.20f, 0.35f, 0.25f,
        0.12f, -0.08f, 0.18f,
    };
    std::vector<float> weight_values = {
        0.20f, -0.10f, 0.05f,
        -0.30f, 0.25f, 0.10f,
        0.15f, 0.05f, -0.20f,
        0.40f, -0.35f, 0.12f,
        -0.05f, 0.18f, 0.30f,
    };
    std::vector<int32_t> labels_values = {
        -100, 2, -100, 1,
        -100, -100, 4, -100,
    };

    auto labels = std::make_shared<Tensor>(std::vector<int64_t>{B, S}, labels_values.data(), kInt32, kCPU);

    auto run_pair = [&](const std::string& reduction) {
        auto weight_dense = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
        auto hidden_dense = std::make_shared<Tensor>(std::vector<int64_t>{B, S, H}, hidden_values.data(), kFloat32, kCPU);
        hidden_dense->set_requires_grad(true);
        weight_dense->set_requires_grad(true);
        auto logits = matmul_rhs_T(hidden_dense, weight_dense);
        auto dense_loss = lm_cross_entropy(logits, labels, -100, reduction);
        dense_loss->backward();

        auto weight_selected = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
        auto hidden_selected = std::make_shared<Tensor>(std::vector<int64_t>{B, S, H}, hidden_values.data(), kFloat32, kCPU);
        hidden_selected->set_requires_grad(true);
        weight_selected->set_requires_grad(true);
        auto selected_loss = selected_token_lm_cross_entropy(hidden_selected, weight_selected, labels, -100, reduction);
        selected_loss->backward();

        auto weight_streaming = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
        auto hidden_streaming = std::make_shared<Tensor>(std::vector<int64_t>{B, S, H}, hidden_values.data(), kFloat32, kCPU);
        hidden_streaming->set_requires_grad(true);
        weight_streaming->set_requires_grad(true);
        auto streaming_loss = streaming_lm_cross_entropy(hidden_streaming, weight_streaming, labels, -100, reduction);
        streaming_loss->backward();

        require_close(selected_loss->data<float>()[0], dense_loss->data<float>()[0],
                      1e-5f, reduction + " loss");
        require_close(streaming_loss->data<float>()[0], dense_loss->data<float>()[0],
                      1e-5f, reduction + " streaming loss");
        require_tensor_close(hidden_selected->grad(), hidden_dense->grad(),
                             1e-5f, reduction + " hidden grad");
        require_tensor_close(hidden_streaming->grad(), hidden_dense->grad(),
                             1e-5f, reduction + " streaming hidden grad");
        require_tensor_close(weight_selected->grad(), weight_dense->grad(),
                             1e-5f, reduction + " weight grad");
        require_tensor_close(weight_streaming->grad(), weight_dense->grad(),
                             1e-5f, reduction + " streaming weight grad");

        std::cout << "[" << reduction << "] dense_loss=" << dense_loss->data<float>()[0]
                  << " selected_loss=" << selected_loss->data<float>()[0]
                  << " streaming_loss=" << streaming_loss->data<float>()[0]
                  << " hidden_grad_diff=" << max_abs_diff(hidden_dense->grad(), hidden_selected->grad())
                  << " weight_grad_diff=" << max_abs_diff(weight_dense->grad(), weight_selected->grad())
                  << std::endl;
    };

    run_pair("mean");
    run_pair("sum");

    std::vector<int32_t> ignored_values(static_cast<size_t>(B * S), -100);
    auto ignored_labels = std::make_shared<Tensor>(std::vector<int64_t>{B, S}, ignored_values.data(), kInt32, kCPU);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    auto hidden_ignored = std::make_shared<Tensor>(std::vector<int64_t>{B, S, H}, hidden_values.data(), kFloat32, kCPU);
    hidden_ignored->set_requires_grad(true);
    auto ignored_loss = selected_token_lm_cross_entropy(hidden_ignored, weight, ignored_labels, -100, "mean");
    ignored_loss->backward();

    if (!std::isnan(ignored_loss->data<float>()[0])) {
        std::cerr << "all-ignored mean selected-token loss should be NaN" << std::endl;
        return 1;
    }
    const float* ignored_grad = hidden_ignored->grad()->data<float>();
    for (int64_t i = 0; i < hidden_ignored->grad()->numel(); ++i) {
        if (std::fabs(ignored_grad[i]) > 1e-7f) {
            std::cerr << "all-ignored selected-token gradient should be zero" << std::endl;
            return 1;
        }
    }

    std::vector<int32_t> invalid_values = labels_values;
    invalid_values[1] = static_cast<int32_t>(V);
    auto invalid_labels = std::make_shared<Tensor>(std::vector<int64_t>{B, S}, invalid_values.data(), kInt32, kCPU);
    bool selected_threw = false;
    try {
        auto hidden_invalid = std::make_shared<Tensor>(std::vector<int64_t>{B, S, H}, hidden_values.data(), kFloat32, kCPU);
        auto invalid_loss = selected_token_lm_cross_entropy(hidden_invalid, weight, invalid_labels, -100, "mean");
        (void)invalid_loss;
    } catch (const std::runtime_error&) {
        selected_threw = true;
    }
    if (!selected_threw) {
        std::cerr << "selected-token CE should reject out-of-range labels" << std::endl;
        return 1;
    }

    std::cout << "[PASS] selected-token LM loss matches dense CE" << std::endl;
    return 0;
}
