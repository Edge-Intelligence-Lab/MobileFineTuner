#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/autograd_engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ops;

namespace {
struct Result { std::vector<float> y, dx, dw; };

std::vector<float> values(const TensorPtr& tensor) {
    if (!tensor) throw std::runtime_error("missing tensor/gradient");
    return {tensor->data<float>(), tensor->data<float>() + tensor->numel()};
}

Result run(bool fast, int batch, bool train_weight, bool transposed_input) {
    autograd::Engine::instance().clear_graph();
    auto base = zeros(transposed_input ? std::vector<int64_t>{batch, 7, 5}
                                      : std::vector<int64_t>{batch, 5, 7});
    auto weight = zeros({13, 7});
    for (int64_t i = 0; i < base->numel(); ++i) base->data<float>()[i] = std::sin(i * 0.17f);
    for (int64_t i = 0; i < weight->numel(); ++i) weight->data<float>()[i] = std::cos(i * 0.23f);
    base->set_requires_grad(true);
    weight->set_requires_grad(train_weight);
    auto input = transposed_input ? transpose(base, 1, 2) : base;
    auto output = fast ? matmul_rhs_T(input, weight) : matmul(input, transpose(weight, 0, 1));
    auto upstream = zeros(output->shape());
    for (int64_t i = 0; i < upstream->numel(); ++i) upstream->data<float>()[i] = std::sin(i * 0.31f);
    output->backward(upstream);
    return {values(output), values(base->grad()), train_weight ? values(weight->grad()) : std::vector<float>{}};
}

double compare(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("shape mismatch");
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double error = std::abs(double(a[i]) - b[i]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || error > 2e-5 + 2e-5 * std::abs(a[i]))
            throw std::runtime_error("head parity failed");
    }
    return maximum;
}
} // namespace

int main() {
    int cases = 0;
    for (int batch : {1, 2}) for (bool train_weight : {false, true}) for (bool transposed : {false, true}) {
        const auto reference = run(false, batch, train_weight, transposed);
        const auto direct = run(true, batch, train_weight, transposed);
        std::cout << "batch=" << batch << " train_weight=" << train_weight << " transposed=" << transposed
                  << " y_error=" << compare(reference.y, direct.y)
                  << " dx_error=" << compare(reference.dx, direct.dx)
                  << " dw_error=" << compare(reference.dw, direct.dw) << '\n';
        ++cases;
    }
    std::cout << "PASS " << cases << " output-head forward/backward cases\n";
}
