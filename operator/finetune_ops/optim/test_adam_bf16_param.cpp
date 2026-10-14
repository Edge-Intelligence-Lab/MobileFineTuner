#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "adam.h"
#include "../core/tensor.h"
#include "../core/dtype.h"

using namespace ops;

int main() {
    try {
        auto param = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kBFloat16, kCPU);

        auto* p = param->data<uint16_t>();
        p[0] = float32_to_bf16_bits(1.0f);
        p[1] = float32_to_bf16_bits(2.0f);
        p[2] = float32_to_bf16_bits(3.0f);
        p[3] = float32_to_bf16_bits(4.0f);

        param->set_requires_grad(true);

        auto grad = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kFloat32, kCPU);

        float* g = grad->data<float>();
        g[0] = 1.0f;
        g[1] = 2.0f;
        g[2] = 3.0f;
        g[3] = 4.0f;

        AdamConfig cfg;
        cfg.learning_rate = 0.1f;
        cfg.beta1 = 0.9f;
        cfg.beta2 = 0.999f;
        cfg.epsilon = 1e-8f;
        cfg.weight_decay = 0.0f;
        cfg.amsgrad = false;

        Adam adam(cfg);

        std::vector<TensorPtr> parameters{param};
        std::vector<TensorPtr> gradients{grad};

        adam.step(parameters, gradients);

        std::cout << "dtype=" << DTypeUtils::to_string(param->dtype()) << "\n";
        std::cout << "updated:";
        for (int i = 0; i < 4; ++i) {
            std::cout << " "
                      << bf16_bits_to_float32(p[i]);
        }
        std::cout << "\n";

        // Step 1 of Adam with positive gradients should move every value down.
        bool moved = true;
        const float expected_before[] = {1.0f, 2.0f, 3.0f, 4.0f};

        for (int i = 0; i < 4; ++i) {
            float v = bf16_bits_to_float32(p[i]);
            if (!(v < expected_before[i])) {
                moved = false;
            }
        }

        if (!moved) {
            std::cerr << "[FAIL] BF16 parameters were not updated downward.\n";
            return 1;
        }

        if (param->dtype() != kBFloat16) {
            std::cerr << "[FAIL] Parameter dtype changed unexpectedly.\n";
            return 2;
        }

        std::cout << "[PASS] BF16 parameter updated by FP32 Adam math.\n";
        std::cout << "[PASS] Parameter storage remains BF16.\n";
        std::cout << "[PASS] This test uses 4 elements only.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 3;
    }
}
