#include "gpt2_model.h"
#include "lora_injector.h"

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace ops;

namespace {

float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    double nd = 0.0;
    double nb = 0.0;
    for (int64_t i = 0; i < a->numel(); ++i) {
        double da = static_cast<double>(pa[i]);
        double db = static_cast<double>(pb[i]);
        nd += (da - db) * (da - db);
        nb += db * db;
    }
    if (nb == 0.0) {
        return static_cast<float>(std::sqrt(nd));
    }
    return static_cast<float>(std::sqrt(nd / nb));
}

void fill_lora_params(const std::vector<TensorPtr>& params) {
    for (size_t p = 0; p < params.size(); ++p) {
        float* data = params[p]->data<float>();
        for (int64_t i = 0; i < params[p]->numel(); ++i) {
            data[i] = 0.01f * static_cast<float>((static_cast<int>(p) + 3 * static_cast<int>(i)) % 17 - 8);
        }
    }
}

}  // namespace

int main() {
    GPT2Config cfg;
    cfg.vocab_size = 64;
    cfg.n_positions = 16;
    cfg.n_embd = 8;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    cfg.use_memory_efficient_attention = false;

    LoraSpec spec;
    spec.rank = 2;
    spec.alpha = 8.0f;
    spec.dropout = 0.0f;
    spec.split_qkv = false;
    spec.targets = {
        LoraTarget::AttnQKV,
        LoraTarget::AttnProj,
        LoraTarget::MlpFcIn,
        LoraTarget::MlpFcOut,
    };

    GPT2Model model_a(cfg);
    model_a.tie_weights();
    LoraInjector injector_a;
    injector_a.inject(model_a, spec);
    auto params_a = injector_a.get_trainable_params();
    fill_lora_params(params_a);

    auto tmp_path =
        std::filesystem::temp_directory_path() / "mobile_finetuner_lora_roundtrip.safetensors";
    std::filesystem::remove(tmp_path);
    injector_a.save_lora_safetensors(tmp_path.string());

    GPT2Model model_b(cfg);
    model_b.tie_weights();
    LoraInjector injector_b;
    injector_b.inject(model_b, spec);
    injector_b.load_lora_safetensors(tmp_path.string());
    auto params_b = injector_b.get_trainable_params();

    if (params_a.size() != params_b.size()) {
        std::cerr << "LoRA param count mismatch: " << params_a.size() << " vs " << params_b.size()
                  << std::endl;
        std::filesystem::remove(tmp_path);
        return 1;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < params_a.size(); ++i) {
        max_diff = std::max(max_diff, rel_diff(params_a[i], params_b[i]));
    }

    bool spec_ok =
        injector_b.spec().rank == spec.rank &&
        std::abs(injector_b.spec().alpha - spec.alpha) < 1e-6f &&
        injector_b.spec().split_qkv == spec.split_qkv &&
        injector_b.spec().targets == spec.targets;

    std::filesystem::remove(tmp_path);

    std::cout << "[LoRARoundTrip] params=" << params_a.size() << " max_diff=" << max_diff
              << " spec_ok=" << (spec_ok ? "true" : "false") << std::endl;

    bool ok = spec_ok && max_diff < 1e-6f;
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}
