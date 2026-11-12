#include "finetune_ops/core/ops.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "finetune_ops/core/autograd_engine.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ops;

namespace {
using Shape = std::vector<int64_t>;
int checks = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Deliberately independent coordinate decoder; do not use the optimized helpers.
Shape decode(int64_t linear, const Shape& shape) {
    Shape coordinate(shape.size());
    for (size_t d = shape.size(); d-- > 0;) {
        coordinate[d] = linear % shape[d];
        linear /= shape[d];
    }
    return coordinate;
}

int64_t encode(const Shape& coordinate, const Shape& shape) {
    int64_t linear = 0;
    for (size_t d = 0; d < shape.size(); ++d) linear = linear * shape[d] + coordinate[d];
    return linear;
}

int64_t broadcast_index(const Shape& coordinate, const Shape& input) {
    int64_t linear = 0;
    const size_t shift = coordinate.size() - input.size();
    for (size_t d = 0; d < input.size(); ++d)
        linear = linear * input[d] + (input[d] == 1 ? 0 : coordinate[d + shift]);
    return linear;
}

void check_copy(const TensorPtr& input, const TensorPtr& output, const std::vector<int>& axes) {
    Shape expected_shape;
    for (int axis : axes) expected_shape.push_back(input->shape()[axis]);
    require(output->shape() == expected_shape && output->dtype() == input->dtype(), "copy metadata");
    const size_t bytes = DTypeUtils::size_of(input->dtype());
    const auto* src = static_cast<const char*>(input->data_ptr());
    const auto* dst = static_cast<const char*>(output->data_ptr());
    for (int64_t i = 0; i < output->numel(); ++i) {
        const auto out_coordinate = decode(i, expected_shape);
        Shape in_coordinate(axes.size());
        for (size_t d = 0; d < axes.size(); ++d) in_coordinate[axes[d]] = out_coordinate[d];
        require(std::memcmp(dst + i * bytes, src + encode(in_coordinate, input->shape()) * bytes, bytes) == 0,
                "copy bytes differ");
    }
    ++checks;
}

void test_copies() {
    const std::vector<Shape> shapes = {{2, 3}, {2, 3, 4}, {2, 1, 3, 5}, {1, 3, 1}, {0, 3, 4}};
    for (DType dtype : {kFloat32, kFloat16, kBFloat16, kInt32, kInt64, kInt8, kBool}) {
        for (const auto& shape : shapes) {
            auto input = zeros(shape, dtype);
            const size_t bytes = static_cast<size_t>(input->numel()) * DTypeUtils::size_of(dtype);
            auto* data = static_cast<unsigned char*>(input->data_ptr());
            for (size_t i = 0; i < bytes; ++i) data[i] = dtype == kBool ? i % 2 : (i * 37 + 13) % 251;
            std::vector<int> axes(shape.size());
            std::iota(axes.begin(), axes.end(), 0);
            do {
                check_copy(input, permute(input, axes), axes);
                auto negative = axes;
                for (int& axis : negative) axis -= static_cast<int>(shape.size());
                check_copy(input, permute(input, negative), axes);
            } while (std::next_permutation(axes.begin(), axes.end()));
            for (size_t a = 0; a < shape.size(); ++a) for (size_t b = 0; b < shape.size(); ++b) {
                std::iota(axes.begin(), axes.end(), 0);
                std::swap(axes[a], axes[b]);
                check_copy(input, transpose(input, static_cast<int>(a), static_cast<int>(b)), axes);
                check_copy(input, input->transpose(static_cast<int>(a), static_cast<int>(b)), axes);
            }
        }
    }
    check_copy(zeros({}), permute(zeros({}), {}), {});
    bool threw = false;
    try { permute(zeros({2, 3}), {0, 0}); } catch (const TensorError&) { threw = true; }
    require(threw, "invalid permutation accepted");
}

float binary(float a, float b, int operation) {
    switch (operation) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        default: return a / b;
    }
}

TensorPtr apply(const TensorPtr& a, const TensorPtr& b, int operation) {
    switch (operation) {
        case 0: return add(a, b);
        case 1: return sub(a, b);
        case 2: return mul(a, b);
        default: return div(a, b);
    }
}

void test_broadcast() {
    const std::vector<std::pair<Shape, Shape>> shapes = {
        {{2, 3, 5}, {5}}, {{2, 3, 5}, {2, 3, 1}}, {{2, 1, 5}, {1, 3, 1}},
        {{2, 3, 1}, {1, 3, 1}}, {{3, 1}, {5}}, {{1}, {2, 3, 5}},
        {{2, 3, 5}, {2, 3, 5}}, {{1, 2, 1, 3, 5}, {1, 4, 3, 1}},
        {{0, 3}, {0, 3}}, {{0, 3}, {1, 3}}, {{2, 0}, {2, 1}}
    };
    for (auto pair : shapes) for (int reverse = 0; reverse < 2; ++reverse) {
        if (reverse) std::swap(pair.first, pair.second);
        for (int operation = 0; operation < 4; ++operation) {
#ifdef USE_NEW_AUTOGRAD_ENGINE
            autograd::Engine::instance().clear_graph();
#endif
            auto a = zeros(pair.first), b = zeros(pair.second);
            for (int64_t i = 0; i < a->numel(); ++i) a->data<float>()[i] = 0.3f + (i % 13) * 0.1f;
            for (int64_t i = 0; i < b->numel(); ++i) b->data<float>()[i] = 0.7f + (i % 11) * 0.2f;
            // Check the unchanged addition/multiplication reductions as well as forward values.
            const bool gradients = (operation == 0 || operation == 2) && a->numel() > 0 && b->numel() > 0;
#ifdef USE_NEW_AUTOGRAD_ENGINE
            a->set_requires_grad(gradients);
            b->set_requires_grad(gradients);
#endif
            auto output = apply(a, b, operation);
            Shape expected_shape(std::max(pair.first.size(), pair.second.size()), 1);
            for (size_t d = 0; d < expected_shape.size(); ++d) {
                const auto da = d < pair.first.size() ? pair.first[pair.first.size() - 1 - d] : 1;
                const auto db = d < pair.second.size() ? pair.second[pair.second.size() - 1 - d] : 1;
                require(da == db || da == 1 || db == 1, "invalid test input shape");
                expected_shape[expected_shape.size() - 1 - d] = da == 1 ? db : da;
            }
            require(output->shape() == expected_shape, "broadcast shape");
            auto upstream = zeros(output->shape());
            std::vector<float> da(a->numel(), 0), db(b->numel(), 0);
            for (int64_t i = 0; i < output->numel(); ++i) {
                const auto coordinate = decode(i, expected_shape);
                const auto ia = broadcast_index(coordinate, pair.first);
                const auto ib = broadcast_index(coordinate, pair.second);
                const float av = a->data<float>()[ia], bv = b->data<float>()[ib];
                require(output->data<float>()[i] == binary(av, bv, operation), "broadcast value");
                const float g = 0.1f + (i % 7) * 0.125f;
                upstream->data<float>()[i] = g;
                da[ia] += g * (operation == 2 ? bv : 1.0f);
                db[ib] += g * (operation == 2 ? av : 1.0f);
            }
            ++checks;
            if (gradients) {
#ifdef USE_NEW_AUTOGRAD_ENGINE
                output->backward(upstream);
                for (const auto& item : {std::make_pair(a, da), std::make_pair(b, db)}) {
                    require(item.first->grad() != nullptr, "missing broadcast gradient");
                    require(item.first->grad()->shape() == item.first->shape(), "gradient shape");
                    for (size_t i = 0; i < item.second.size(); ++i)
                        require(std::abs(item.first->grad()->data<float>()[i] - item.second[i]) < 2e-5f,
                                "broadcast gradient");
                }
                ++checks;
#endif
            }
        }
    }
}
} // namespace

int main() {
    test_copies();
    test_broadcast();
    std::cout << "PASS " << checks << " copy/broadcast checks\n";
}
