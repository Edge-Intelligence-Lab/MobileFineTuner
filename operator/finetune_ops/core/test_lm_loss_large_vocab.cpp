#include "autograd_engine.h"
#include "lm_loss.h"
#include "ops.h"
#include <cmath>
#include <iostream>
using namespace ops;
void close(double got, double expected, double tolerance, const char* message) {
    if (!std::isfinite(got) || std::abs(got - expected) > tolerance)
        throw std::runtime_error(message);
}
int main() try {
    constexpr int V = 248320;
    const double probability = 1.0 / (1.0 + (V - 1) * std::exp(-12.0));
    auto labels = zeros({1, 2}, kInt32);
    auto logits = zeros({1, 2, V});
    logits->data<float>()[0] = 12;
    logits->set_requires_grad(true);
    auto loss = lm_cross_entropy(logits, labels, -100, "mean");
    loss->backward();
    close(loss->data<float>()[0], -std::log(probability), 2e-7, "large-vocabulary loss");
    close(logits->grad()->data<float>()[0], probability - 1, 2e-7,
          "large-vocabulary target gradient");
    double shift_gradient = 0;
    for (int i = 0; i < V; ++i)
        shift_gradient += logits->grad()->data<float>()[i];
    close(shift_gradient, 0, 3e-7, "softmax gradient must preserve additive-shift invariance");
    autograd::Engine::instance().clear_graph();
    auto hidden = ones({1, 2, 1});
    hidden->set_requires_grad(true);
    auto weights = zeros({V, 1});
    weights->data<float>()[0] = 12;
    auto streaming = streaming_lm_cross_entropy(hidden, weights, labels, -100, "mean");
    streaming->backward();
    close(streaming->data<float>()[0], -std::log(probability), 2e-7,
          "streaming large-vocabulary loss");
    close(hidden->grad()->data<float>()[0], 12 * (probability - 1), 2e-6,
          "streaming large-vocabulary gradient");
    std::cout << "Large-vocabulary dense/streaming loss and shift-invariant "
                 "gradients PASS\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
