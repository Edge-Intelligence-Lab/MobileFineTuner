#include "../core/ops.h"
#include "adam.h"
#include <cmath>
#include <iostream>
#include <limits>
using namespace ops;
int main() try {
    auto parameter = zeros({1000000});
    Adam optimizer{AdamConfig{}};
    for (float magnitude : {1e-3f, 1e20f}) {
        auto gradient = full(parameter->shape(), magnitude);
        parameter->set_grad(gradient);
        optimizer.clip_grad_norm({parameter}, 0.5f);
        double squared = 0;
        for (int64_t i = 0; i < gradient->numel(); ++i) {
            const double value = gradient->data<float>()[i];
            squared += value * value;
        }
        if (std::abs(std::sqrt(squared) - 0.5) > 2e-6)
            throw std::runtime_error("High-dimensional gradient clipping is inaccurate");
    }
    parameter->grad()->data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try {
        optimizer.clip_grad_norm({parameter}, 1.0f);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Non-finite gradient norm was not rejected");
    std::cout
        << "PASS: gradient clipping uses accurate accumulation and rejects non-finite gradients\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
