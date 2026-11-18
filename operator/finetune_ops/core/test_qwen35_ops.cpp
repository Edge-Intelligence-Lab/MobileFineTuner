#include "autograd_engine.h"
#include "ops.h"
#include "qwen35_ops.h"
#include <cmath>
#include <functional>
#include <iostream>
using namespace ops;
void check_grad(const std::vector<TensorPtr>& inputs, const std::function<TensorPtr()>& forward,
                const char* name) {
    auto& engine = autograd::Engine::instance();
    engine.clear_graph();
    engine.set_enabled(true);
    for (auto& p : inputs)
        p->set_requires_grad(true);
    auto y = forward();
    auto seed = zeros(y->shape());
    for (int64_t i = 0; i < seed->numel(); ++i)
        seed->data<float>()[i] = std::sin(float(i + 1) * 0.7f);
    auto loss = sum(mul(y, seed));
    loss->backward();
    std::vector<TensorPtr> grads;
    for (auto& p : inputs) {
        if (!p->grad())
            throw std::runtime_error("missing grad");
        grads.push_back(p->grad());
    }
    engine.clear_graph();
    engine.set_enabled(false);
    double worst = 0;
    auto evaluate = [&]() {
        auto o = forward();
        double v = 0;
        for (int64_t i = 0; i < o->numel(); ++i)
            v += double(o->data<float>()[i]) * seed->data<float>()[i];
        return v;
    };
    for (size_t p = 0; p < inputs.size(); ++p)
        for (int64_t i = 0; i < inputs[p]->numel(); ++i) {
            auto& value = inputs[p]->data<float>()[i];
            const float original = value, eps = 1e-3f;
            value = original + eps;
            double plus = evaluate();
            value = original - eps;
            double minus = evaluate();
            value = original;
            double fd = (plus - minus) / (2 * eps), analytic = grads[p]->data<float>()[i];
            worst = std::max(worst, std::abs(fd - analytic));
            if (!std::isfinite(analytic) || std::abs(fd - analytic) > 5e-4 + 3e-3 * std::abs(fd))
                throw std::runtime_error(
                    std::string(name) + " derivative mismatch input=" + std::to_string(p) +
                    " element=" + std::to_string(i) + " analytic=" + std::to_string(analytic) +
                    " finite_difference=" + std::to_string(fd));
        }
    engine.set_enabled(true);
    std::cout << name << " max_abs=" << worst << " PASS\n";
}
TensorPtr values(std::vector<int64_t> shape, float offset = 0) {
    auto t = zeros(shape);
    for (int64_t i = 0; i < t->numel(); ++i)
        t->data<float>()[i] = 0.3f * std::sin(float(i + 1)) + offset;
    return t;
}
int main() try {
    auto x = values({2, 5, 3}), w = values({3, 1, 4});
    check_grad({x, w}, [&] { return causal_depthwise_conv1d(x, w); }, "causal convolution");
    auto r = values({2, 2, 5, 8});
    check_grad({r}, [&] { return partial_rotary_embedding(r, 4, 10000); }, "partial RoPE");
    for (int heads : {1, 2}) {
        auto q = values({2, 5, 1, 3}, 0.2f), k = values({2, 5, 1, 3}, 0.3f),
             v = values({2, 5, heads, 2});
        auto a = values({2, 5, heads}, -0.4f), b = values({2, 5, heads}),
             logA = values({heads}, -0.7f), dt = values({heads});
        check_grad(
            {q, k, v, a, b, logA, dt}, [&] { return gated_delta_rule(q, k, v, a, b, logA, dt); },
            "gated delta recurrence");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
