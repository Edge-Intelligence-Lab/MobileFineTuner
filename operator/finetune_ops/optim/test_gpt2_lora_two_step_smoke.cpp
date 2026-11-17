#include "../core/autograd_engine.h"
#include "../core/lm_loss.h"
#include "../core/memory_manager.h"
#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "adam.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace ops;

namespace {

void fill_constant(const TensorPtr& tensor, float value) {
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = value;
    }
}

void fill_uniform(const TensorPtr& tensor, float low, float high, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(low, high);
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = dist(rng);
    }
}

void initialize_tiny_gpt2(GPT2Model& model) {
    std::mt19937 rng(7);
    auto params = model.parameters();
    size_t idx = 0;

    fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // wte
    fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // wpe

    const int layers = model.config().n_layer;
    for (int layer = 0; layer < layers; ++layer) {
        fill_constant(params[idx++], 1.0f);            // ln_1 weight
        fill_constant(params[idx++], 0.0f);            // ln_1 bias
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // attn qkv weight
        fill_constant(params[idx++], 0.0f);            // attn qkv bias
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // attn proj weight
        fill_constant(params[idx++], 0.0f);            // attn proj bias
        fill_constant(params[idx++], 1.0f);            // ln_2 weight
        fill_constant(params[idx++], 0.0f);            // ln_2 bias
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // mlp fc_in weight
        fill_constant(params[idx++], 0.0f);            // mlp fc_in bias
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);  // mlp fc_out weight
        fill_constant(params[idx++], 0.0f);            // mlp fc_out bias
    }

    fill_constant(params[idx++], 1.0f);                // ln_f weight
    fill_constant(params[idx++], 0.0f);                // ln_f bias
}

TensorPtr make_input_ids() {
    auto ids = std::make_shared<Tensor>(std::vector<int64_t>{2, 6}, kInt32, kCPU);
    int32_t values[] = {
        1, 2, 3, 4, 5, 6,
        2, 3, 4, 5, 6, 7,
    };
    std::memcpy(ids->data<int32_t>(), values, sizeof(values));
    return ids;
}

TensorPtr make_labels() {
    auto labels = std::make_shared<Tensor>(std::vector<int64_t>{2, 6}, kInt32, kCPU);
    int32_t values[] = {
        2, 3, 4, 5, 6, 7,
        3, 4, 5, 6, 7, 8,
    };
    std::memcpy(labels->data<int32_t>(), values, sizeof(values));
    return labels;
}

double grad_l2_norm(const std::vector<TensorPtr>& grads) {
    double accum = 0.0;
    for (const auto& grad : grads) {
        if (!grad) continue;
        const float* data = grad->data<float>();
        for (int64_t i = 0; i < grad->numel(); ++i) {
            accum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
        }
    }
    return std::sqrt(accum);
}

double max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* da = a->data<float>();
    const float* db = b->data<float>();
    double max_diff = 0.0;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(da[i]) - static_cast<double>(db[i])));
    }
    return max_diff;
}

bool is_finite_scalar(const TensorPtr& tensor) {
    return tensor && tensor->numel() == 1 && std::isfinite(static_cast<double>(tensor->data<float>()[0]));
}

}  // namespace

int run_case(bool quantized, bool streaming, bool split) {
    autograd::Engine::instance().clear_graph();
    GPT2Config cfg;
    cfg.vocab_size = 64;
    cfg.n_positions = 16;
    cfg.n_embd = 32;
    cfg.n_layer = 1;
    cfg.n_head = 4;
    cfg.use_memory_efficient_attention = streaming;

    GPT2Model model(cfg);
    initialize_tiny_gpt2(model);
    model.tie_weights();

    LoraSpec spec;
    spec.rank = 2;
    spec.alpha = 8.0f;
    spec.dropout = 0.0f;
    spec.split_qkv = split;
    spec.targets = {LoraTarget::AttnQKV, LoraTarget::AttnProj};

    LoraInjector injector;
    injector.inject(model, spec);
    if (quantized) {
        auto& block = model.get_block(0);
        for (auto* layer : {block.qkv_lin.get(), block.proj_lin.get(), block.fc_in_lin.get(),
                            block.fc_out_lin.get()}) {
            layer->quantize_base_weight_q8_0();
        }
    }
    auto lora_params = injector.get_trainable_params();
    if (lora_params.empty()) {
        std::cerr << "LoRA parameter list is empty" << std::endl;
        return 1;
    }

    std::vector<TensorPtr> before;
    before.reserve(lora_params.size());
    for (const auto& param : lora_params) {
        before.push_back(std::make_shared<Tensor>(*param));
    }
    auto input_ids = make_input_ids();
    auto labels = make_labels();

    AdamConfig opt_cfg;
    opt_cfg.learning_rate = 5e-2f;
    opt_cfg.weight_decay = 0.0f;
    Adam optimizer(opt_cfg);

    std::vector<float> losses;
    for (int step = 0; step < 2; ++step) {
        auto logits = model.forward(input_ids);
        auto loss = lm_cross_entropy(logits, labels, -100, "mean");
        if (!is_finite_scalar(loss)) {
            std::cerr << "Non-finite loss at step " << step << std::endl;
            return 1;
        }
        losses.push_back(loss->data<float>()[0]);

        loss->backward();

        // An aggregate gradient/update check can pass while the entire QKV
        // branch is disconnected. Each Q/K/V part of the fused B adapter must
        // receive a nonzero gradient, including on the first (zero-B) step.
        const auto& slices = model.get_block(0).qkv_lin->slices();
        for (int part = 0; part < 3; ++part) {
            const auto gradient = slices[split ? part : 0].B->grad();
            if (!gradient) {
                std::cerr << "Missing Q/K/V gradient" << std::endl;
                return 1;
            }
            double norm_sq = 0;
            const int width = split ? cfg.n_embd : 3 * cfg.n_embd;
            for (int r = 0; r < spec.rank; ++r)
                for (int c = 0; c < cfg.n_embd; ++c) {
                    const double g =
                        gradient->data<float>()[r * width + (split ? 0 : part * cfg.n_embd) + c];
                    norm_sq += g * g;
                }
            if (!(norm_sq > 0) || !std::isfinite(norm_sq)) {
                std::cerr << "Invalid Q/K/V gradient" << std::endl;
                return 1;
            }
        }

        std::vector<TensorPtr> grads;
        grads.reserve(lora_params.size());
        for (const auto& param : lora_params) {
            grads.push_back(param->grad());
        }

        double grad_norm = grad_l2_norm(grads);
        if (!(grad_norm > 0.0) || !std::isfinite(grad_norm)) {
            std::cerr << "Invalid gradient norm at step " << step << ": " << grad_norm << std::endl;
            return 1;
        }

        optimizer.step(lora_params, grads);
        for (auto& param : lora_params) {
            param->zero_grad();
        }
        MemoryManager::instance().force_cleanup();
    }

    double param_delta = 0.0;
    for (size_t i = 0; i < lora_params.size(); ++i) {
        param_delta = std::max(param_delta, max_abs_diff(before[i], lora_params[i]));
    }
    std::cout << "[GPT2LoRATwoStepSmoke] loss0=" << losses[0] << " loss1=" << losses[1]
              << " param_delta=" << param_delta << " quantized=" << quantized
              << " streaming=" << streaming << " split=" << split << std::endl;

    bool ok =
        std::isfinite(static_cast<double>(losses[0])) &&
        std::isfinite(static_cast<double>(losses[1])) &&
        param_delta > 0.0;
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}

int main() {
    for (bool quantized : {false, true}) {
        for (bool streaming : {false, true}) {
            for (bool split : {false, true})
                if (run_case(quantized, streaming, split) != 0)
                    return 1;
        }
    }
    return 0;
}
