#include "../core/autograd_engine.h"
#include "../core/ops.h"
#include "lora_linear.h"

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
    if (t->numel() != static_cast<int64_t>(values.size())) {
        std::cerr << "value count mismatch" << std::endl;
        std::exit(1);
    }
    std::memcpy(t->data<float>(), values.data(), values.size() * sizeof(float));
    return t;
}

TensorPtr make_bf16(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    auto t = std::make_shared<Tensor>(shape, kBFloat16, kCPU);
    if (t->numel() != static_cast<int64_t>(values.size())) {
        std::cerr << "value count mismatch" << std::endl;
        std::exit(1);
    }
    for (int64_t i = 0; i < t->numel(); ++i) {
        t->data<uint16_t>()[i] = float32_to_bf16_bits(values[static_cast<size_t>(i)]);
    }
    return t;
}

float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(pa[i] - pb[i]));
    }
    return max_diff;
}

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

void test_fp32_subcolumn_merge_matches_forward() {
    auto W = make_f32({3, 4}, {
        0.10f, -0.20f, 0.30f, 0.40f,
        -0.50f, 0.60f, -0.70f, 0.80f,
        0.90f, -1.00f, 1.10f, -1.20f,
    });
    auto A = make_f32({3, 2}, {
        0.10f, -0.20f,
        0.30f, 0.40f,
        -0.50f, 0.60f,
    });
    auto B = make_f32({2, 2}, {
        0.70f, -0.80f,
        0.90f, 1.00f,
    });
    auto x = make_f32({1, 2, 3}, {
        1.0f, -2.0f, 0.5f,
        -1.5f, 0.25f, 2.0f,
    });

    LoRALinear layer(W);
    layer.attach_lora(A, B, 0.5f, 1, 2);

    auto before = layer.forward(x);
    layer.merge_to_base();
    require(layer.is_merged(), "LoRA layer did not enter merged state");
    auto after_merge = layer.forward(x);

    float diff = max_abs_diff(before, after_merge);
    require(diff < 1e-6f, "merged FP32 subcolumn forward does not match unmerged forward");

    layer.unmerge_from_base();
    require(!layer.is_merged(), "LoRA layer did not leave merged state");
    auto after_unmerge = layer.forward(x);
    diff = max_abs_diff(before, after_unmerge);
    require(diff < 1e-6f, "unmerged FP32 forward changed after round trip");
}

void test_bf16_merge_restores_original_storage() {
    auto W = make_bf16({2, 3}, {
        0.10f, -0.20f, 0.30f,
        0.40f, -0.50f, 0.60f,
    });
    auto A = make_f32({2, 2}, {
        0.25f, -0.50f,
        0.75f, 0.125f,
    });
    auto B = make_f32({2, 2}, {
        0.40f, -0.30f,
        0.20f, 0.10f,
    });

    std::vector<uint16_t> original(W->data<uint16_t>(), W->data<uint16_t>() + W->numel());

    LoRALinear layer(W);
    layer.attach_lora(A, B, 1.0f, 1, 2);
    layer.merge_to_base();
    require(layer.is_merged(), "BF16 LoRA merge did not set merged state");

    bool changed = false;
    for (int64_t i = 0; i < W->numel(); ++i) {
        changed = changed || W->data<uint16_t>()[i] != original[static_cast<size_t>(i)];
    }
    require(changed, "BF16 LoRA merge did not modify base storage");

    layer.unmerge_from_base();
    require(!layer.is_merged(), "BF16 LoRA unmerge did not clear merged state");
    for (int64_t i = 0; i < W->numel(); ++i) {
        require(W->data<uint16_t>()[i] == original[static_cast<size_t>(i)],
                "BF16 LoRA unmerge failed to restore original bytes");
    }
}

void test_q8_subnormal_scale_matches_fp32() {
    auto weight = make_f32({64, 32}, std::vector<float>(64 * 32, 1e-4f));
    auto input = make_f32({1, 1, 64}, std::vector<float>(64, 1.0f));
    LoRALinear layer(weight);
    auto expected = layer.forward(input);
    layer.quantize_base_weight_q8_0();
    auto quantized = layer.forward(input);
    const float error = max_abs_diff(expected, quantized);
    require(error < 0.02f * std::fabs(expected->data<float>()[0]),
            "Q8 FP16 subnormal scales incorrectly halve the base output");
}

void test_clear_lora_unmerges_first() {
    auto W = make_bf16({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto A = make_f32({2, 1}, {0.5f, -0.25f});
    auto B = make_f32({1, 2}, {0.75f, -0.5f});
    std::vector<uint16_t> original(W->data<uint16_t>(), W->data<uint16_t>() + W->numel());

    LoRALinear layer(W);
    layer.attach_lora(A, B, 1.0f);
    layer.merge_to_base();
    layer.clear_lora();

    require(!layer.is_merged(), "clear_lora left layer in merged state");
    require(layer.slices().empty(), "clear_lora did not remove slices");
    for (int64_t i = 0; i < W->numel(); ++i) {
        require(W->data<uint16_t>()[i] == original[static_cast<size_t>(i)],
                "clear_lora failed to restore merged base weight");
    }
}

#ifdef USE_NEW_AUTOGRAD_ENGINE
void test_subcolumn_backward_finite_difference() {
    auto& engine = autograd::Engine::instance();
    engine.clear_graph();
    engine.set_enabled(true);
    auto weight = zeros({3, 5});
    for (int i = 0; i < 15; ++i)
        weight->data<float>()[i] = 0.1f * std::sin(float(i + 1));
    auto a = make_f32({3, 2}, {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f});
    auto b = make_f32({2, 2}, {0.7f, -0.8f, 0.9f, 1.0f});
    auto x = make_f32({1, 2, 3}, {1.0f, -2.0f, 0.5f, -1.5f, 0.25f, 2.0f});
    x->set_requires_grad(true);
    LoRALinear layer(weight);
    layer.attach_lora(a, b, 0.5f, 2, 2);
    auto seed = ones({1, 2, 5});
    for (int i = 0; i < 10; ++i)
        seed->data<float>()[i] = 0.1f * (i - 3);
    auto loss = sum(mul(layer.forward(x), seed));
    loss->backward();
    const std::vector<TensorPtr> inputs = {x, a, b};
    std::vector<TensorPtr> gradients;
    for (const auto& p : inputs) {
        require(bool(p->grad()), "Subcolumn gradient missing");
        gradients.push_back(p->grad()->clone());
    }
    engine.clear_graph();
    engine.set_enabled(false);
    auto evaluate = [&] {
        auto output = layer.forward(x);
        double value = 0;
        for (int i = 0; i < 10; ++i)
            value += double(output->data<float>()[i]) * seed->data<float>()[i];
        return value;
    };
    for (size_t p = 0; p < inputs.size(); ++p)
        for (int64_t i = 0; i < inputs[p]->numel(); ++i) {
            auto& value = inputs[p]->data<float>()[i];
            const float original = value, eps = 1e-3f;
            value = original + eps;
            const double plus = evaluate();
            value = original - eps;
            const double minus = evaluate();
            value = original;
            const double reference = (plus - minus) / (2 * eps),
                         actual = gradients[p]->data<float>()[i];
            require(std::abs(reference - actual) < 5e-4 + 1e-3 * std::abs(reference),
                    "Subcolumn derivative failed finite differences");
        }
    engine.set_enabled(true);
}
#endif

}  // namespace

int main() {
    test_fp32_subcolumn_merge_matches_forward();
#ifdef USE_NEW_AUTOGRAD_ENGINE
    test_subcolumn_backward_finite_difference();
#endif
    test_bf16_merge_restores_original_storage();
    test_clear_lora_unmerges_first();
    test_q8_subnormal_scale_matches_fp32();
    std::cout << "[PASS] LoRALinear merge/unmerge tests" << std::endl;
    return 0;
}
