#include "ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ops;

namespace {

TensorPtr make_tensor(const std::vector<float>& values) {
    return std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(values.size())},
        values.data(),
        kFloat32,
        kCPU);
}

void require_close(float actual, float expected, const char* name) {
    if (std::fabs(actual - expected) > 1e-6f) {
        std::cerr << name << " mismatch: actual=" << actual
                  << " expected=" << expected << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main() {
    auto p = make_tensor({1.0f, -2.0f, 3.0f});
    p->set_requires_grad(true);

    auto x1 = make_tensor({0.5f, 1.0f, -1.5f});
    auto loss1 = sum(mul(p, x1), -1, false);
    loss1->backward();

    auto x2 = make_tensor({2.0f, -0.25f, 0.75f});
    auto loss2 = sum(mul(p, x2), -1, false);
    loss2->backward();

    const float* g = p->grad()->data<float>();
    require_close(g[0], 2.5f, "accum grad[0]");
    require_close(g[1], 0.75f, "accum grad[1]");
    require_close(g[2], -0.75f, "accum grad[2]");

    p->zero_grad();
    auto loss3 = sum(mul(p, x1), -1, false);
    loss3->backward();
    g = p->grad()->data<float>();
    require_close(g[0], 0.5f, "zeroed grad[0]");
    require_close(g[1], 1.0f, "zeroed grad[1]");
    require_close(g[2], -1.5f, "zeroed grad[2]");

    std::cout << "[PASS] autograd leaf gradients accumulate across backward calls" << std::endl;
    return 0;
}
