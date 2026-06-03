#include "ops.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ops;

int main() {
    constexpr int64_t B = 1;
    constexpr int64_t H = 1;
    constexpr int64_t S = 2;
    constexpr int64_t D = 4;
    constexpr float theta = 10000.0f;
    constexpr float tol = 1e-5f;

    auto x = zeros({B, H, S, D}, kFloat32, kCPU);
    float* data = x->data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;  // pos 0
    data[4] = 1.0f; data[5] = 2.0f; data[6] = 3.0f; data[7] = 4.0f;  // pos 1

    auto y = apply_rope(x, static_cast<int>(S), static_cast<int>(D), theta);
    const float* out = y->data<float>();

    // Position 0 angle is zero, so it should be unchanged.
    for (int i = 0; i < D; ++i) {
        if (std::fabs(out[i] - data[i]) > tol) {
            throw std::runtime_error("position 0 changed unexpectedly");
        }
    }

    const int64_t base = D;  // position 1 offset
    for (int64_t d = 0; d < D / 2; ++d) {
        const float freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(d) / static_cast<float>(D));
        const float angle = freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float x1 = data[base + d];
        const float x2 = data[base + D / 2 + d];
        const float expected_1 = x1 * c - x2 * s;
        const float expected_2 = x2 * c + x1 * s;
        if (std::fabs(out[base + d] - expected_1) > tol) {
            throw std::runtime_error("first-half rope value mismatch");
        }
        if (std::fabs(out[base + D / 2 + d] - expected_2) > tol) {
            throw std::runtime_error("second-half rope value mismatch");
        }
    }

    std::cout << "ApplyRoPE layout test passed\n";
    return 0;
}
