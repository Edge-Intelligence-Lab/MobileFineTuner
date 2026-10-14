#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "../core/ops.h"
#include "../core/tensor.h"
#include "../core/dtype.h"
#include "adam.h"

using namespace ops;

static float read_bf16(const TensorPtr& t, int64_t i) {
    return bf16_bits_to_float32(t->data<uint16_t>()[i]);
}

int main() {
    try {
        // Small FP32 activation.
        auto x = std::make_shared<Tensor>(
            std::vector<int64_t>{2, 3}, kFloat32, kCPU);

        float* xd = x->data<float>();
        xd[0] = 1.0f;  xd[1] = 2.0f;  xd[2] = 3.0f;
        xd[3] = 4.0f;  xd[4] = 5.0f;  xd[5] = 6.0f;

        // Trainable BF16 weight.
        auto w = std::make_shared<Tensor>(
            std::vector<int64_t>{3, 2}, kBFloat16, kCPU);

        uint16_t* wd = w->data<uint16_t>();
        const float initial[] = {
            0.10f, 0.20f,
            0.30f, 0.40f,
            0.50f, 0.60f
        };

        for (int i = 0; i < 6; ++i) {
            wd[i] = float32_to_bf16_bits(initial[i]);
        }

        w->set_requires_grad(true);

        // This is the important operation:
        // FP32 activation × BF16 trainable weight -> FP32 output.
        auto y = matmul(x, w);

        if (y->dtype() != kFloat32) {
            std::cerr << "[FAIL] Matmul output is not FP32.\n";
            return 1;
        }

        // Stable target with same shape.
        auto target = std::make_shared<Tensor>(
            std::vector<int64_t>{2, 2}, kFloat32, kCPU);

        float* td = target->data<float>();
        for (int i = 0; i < 4; ++i) td[i] = 0.0f;

        auto loss = mse_loss(y, target, "mean");

        if (!loss) {
            std::cerr << "[FAIL] Loss creation failed.\n";
            return 2;
        }

        loss->backward();

        auto grad = w->grad();

        if (!grad) {
            std::cerr << "[FAIL] BF16 weight received no gradient.\n";
            return 3;
        }

        if (grad->dtype() != kFloat32) {
            std::cerr << "[FAIL] BF16 weight gradient is not FP32.\n";
            return 4;
        }

        std::cout << "loss=" << std::setprecision(9)
                  << loss->item() << "\n";

        std::cout << "weight_before:";
        for (int i = 0; i < 6; ++i) {
            std::cout << " " << read_bf16(w, i);
        }
        std::cout << "\n";

        AdamConfig cfg;
        cfg.learning_rate = 1e-2f;
        cfg.beta1 = 0.9f;
        cfg.beta2 = 0.999f;
        cfg.epsilon = 1e-8f;
        cfg.weight_decay = 0.0f;
        cfg.amsgrad = false;

        Adam adam(cfg);
        adam.step({w}, {grad});

        std::cout << "weight_after:";
        for (int i = 0; i < 6; ++i) {
            std::cout << " " << read_bf16(w, i);
        }
        std::cout << "\n";

        bool changed = false;

        for (int i = 0; i < 6; ++i) {
            float after = read_bf16(w, i);
            float before =
                bf16_bits_to_float32(
                    float32_to_bf16_bits(initial[i]));

            if (after != before) {
                changed = true;
                break;
            }
        }

        if (!changed) {
            std::cerr << "[FAIL] BF16 weights did not change after Adam.\n";
            return 5;
        }

        if (w->dtype() != kBFloat16) {
            std::cerr << "[FAIL] Weight dtype changed.\n";
            return 6;
        }

        std::cout << "[PASS] FP32 activation × BF16 trainable weight.\n";
        std::cout << "[PASS] BF16 weight received FP32 gradient.\n";
        std::cout << "[PASS] FP32 Adam updated BF16 weight.\n";
        std::cout << "[PASS] Weight storage remains BF16.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 10;
    }
}
