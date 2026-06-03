#include "ops.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ops;

namespace {

TensorPtr make_f32(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    auto t = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    std::memcpy(t->data<float>(), values.data(), values.size() * sizeof(float));
    return t;
}

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

void require_close(float got, float expected, const char* name) {
    if (std::fabs(got - expected) > 1e-6f) {
        std::cerr << name << " got=" << got << " expected=" << expected << std::endl;
        std::exit(1);
    }
}

void test_positive_step_slice_values() {
    auto x = make_f32({2, 5}, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    });
    auto y = x->slice(1, 0, 5, 2);
    require(y->shape() == std::vector<int64_t>({2, 3}), "slice step output shape mismatch");
    const float* data = y->data<float>();
    const std::vector<float> expected = {0.0f, 2.0f, 4.0f, 5.0f, 7.0f, 9.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(data[i], expected[i], "slice step value");
    }
}

void test_positive_step_slice_backward() {
    auto x = make_f32({2, 5}, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    });
    x->set_requires_grad(true);
    auto y = x->slice(1, 1, 5, 2);
    auto loss = sum(y);
    loss->backward();

    const float* grad = x->grad()->data<float>();
    const std::vector<float> expected = {
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(grad[i], expected[i], "slice step grad");
    }
}

void test_non_float_slice_copies_bytes() {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{5}, kInt32, kCPU);
    for (int i = 0; i < 5; ++i) {
        x->data<int32_t>()[i] = i + 1;
    }
    auto y = x->slice(0, 0, 5, 2);
    require(y->shape() == std::vector<int64_t>({3}), "int32 slice step output shape mismatch");
    require(y->data<int32_t>()[0] == 1, "int32 slice value 0 mismatch");
    require(y->data<int32_t>()[1] == 3, "int32 slice value 1 mismatch");
    require(y->data<int32_t>()[2] == 5, "int32 slice value 2 mismatch");
}

}  // namespace

int main() {
    test_positive_step_slice_values();
    test_positive_step_slice_backward();
    test_non_float_slice_copies_bytes();
    std::cout << "[PASS] tensor slice step tests" << std::endl;
    return 0;
}
