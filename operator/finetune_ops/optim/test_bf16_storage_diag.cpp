#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../core/tensor.h"
#include "../core/dtype.h"
#include "adam.h"

using namespace ops;

static void dump(const char* tag, const TensorPtr& t) {
    auto* u = t->data<uint16_t>();
    std::cout << tag << "\n";
    for (int i = 0; i < 4; ++i) {
        float f = bf16_bits_to_float32(u[i]);
        std::cout << "  [" << i << "] bits=0x"
                  << std::hex << std::setw(4) << std::setfill('0') << u[i]
                  << std::dec << " value=" << std::setprecision(9) << f
                  << "\n";
    }
}

int main() {
    try {
        auto t = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kBFloat16, kCPU);

        auto* u = t->data<uint16_t>();
        const float initial[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        for (int i = 0; i < 4; ++i) {
            u[i] = float32_to_bf16_bits(initial[i]);
        }

        dump("BEFORE ADAM:", t);

        auto g = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kFloat32, kCPU);

        float* gd = g->data<float>();
        for (int i = 0; i < 4; ++i) gd[i] = 1.0f;

        AdamConfig cfg;
        cfg.learning_rate = 0.1f;
        cfg.beta1 = 0.9f;
        cfg.beta2 = 0.999f;
        cfg.epsilon = 1e-8f;
        cfg.weight_decay = 0.0f;
        cfg.amsgrad = false;

        Adam adam(cfg);

        t->set_requires_grad(true);
        adam.step({t}, {g});

        dump("AFTER ADAM:", t);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
