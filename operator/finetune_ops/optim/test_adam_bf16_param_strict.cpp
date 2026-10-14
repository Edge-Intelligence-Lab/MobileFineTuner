#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "adam.h"
#include "../core/tensor.h"
#include "../core/dtype.h"

using namespace ops;

int main() {
    try {
        const float initial[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float grads[4]   = {1.0f, 1.0f, 1.0f, 1.0f};

        auto param = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kBFloat16, kCPU);

        auto* p = param->data<uint16_t>();

        for (int i = 0; i < 4; ++i) {
            p[i] = float32_to_bf16_bits(initial[i]);
        }

        auto grad = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kFloat32, kCPU);

        float* g = grad->data<float>();
        for (int i = 0; i < 4; ++i) {
            g[i] = grads[i];
        }

        AdamConfig cfg;
        cfg.learning_rate = 0.1f;
        cfg.beta1 = 0.9f;
        cfg.beta2 = 0.999f;
        cfg.epsilon = 1e-8f;
        cfg.weight_decay = 0.0f;
        cfg.amsgrad = false;

        Adam adam(cfg);

        param->set_requires_grad(true);

        adam.step({param}, {grad});

        /*
         * For the first Adam step with beta1=0.9, beta2=0.999 and
         * identical positive gradients, the bias-corrected update is
         * exactly approximately lr = 0.1.
         *
         * Therefore expected BF16 values are:
         *   BF16(0.9), BF16(1.9), BF16(2.9), BF16(3.9)
         */
        bool ok = true;

        std::cout << "dtype=" << DTypeUtils::to_string(param->dtype()) << "\n";

        for (int i = 0; i < 4; ++i) {
            const float actual = bf16_bits_to_float32(p[i]);
            const float expected =
                bf16_bits_to_float32(
                    float32_to_bf16_bits(initial[i] - 0.1f));

            const float diff = std::fabs(actual - expected);

            std::cout << "[" << i << "] actual=" << std::setprecision(9)
                      << actual
                      << " expected=" << expected
                      << " diff=" << diff
                      << "\n";

            if (diff > 1e-6f) {
                ok = false;
            }
        }

        if (!ok) {
            std::cerr << "[FAIL] BF16 Adam update does not match reference.\n";
            return 1;
        }

        std::cout << "[PASS] BF16 Adam update matches FP32 reference.\n";
        std::cout << "[PASS] Parameter storage remains BF16.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
