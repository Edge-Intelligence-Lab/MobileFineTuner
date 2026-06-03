#include "gemma_lora_injector.h"

#include <algorithm>
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
            data[i] = 0.02f * static_cast<float>((static_cast<int>(p) + 5 * static_cast<int>(i)) % 19 - 9);
        }
    }
}

GemmaTextConfig make_test_config() {
    GemmaTextConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 8;
    cfg.intermediate_size = 16;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 4;
    cfg.max_position_embeddings = 32;
    cfg.sliding_window = 16;
    cfg.layer_types.assign(cfg.num_hidden_layers, "sliding_attention");
    return cfg;
}

}  // namespace

int main() {
    GemmaTextConfig cfg = make_test_config();

    GemmaLoraSpec spec = GemmaLoraSpec::full_attn_mlp();
    spec.rank = 2;
    spec.alpha = 8.0f;
    spec.dropout = 0.0f;

    GemmaModel model_a(cfg);
    GemmaLoraInjector injector_a;
    injector_a.inject(model_a, spec);
    auto params_a = injector_a.get_trainable_params();
    fill_lora_params(params_a);

    auto tmp_path =
        std::filesystem::temp_directory_path() / "mobile_finetuner_gemma_lora_roundtrip.safetensors";
    std::filesystem::remove(tmp_path);
    injector_a.save_lora_safetensors(tmp_path.string());

    GemmaModel model_b(cfg);
    GemmaLoraInjector injector_b;
    injector_b.inject(model_b, spec);
    injector_b.load_lora_safetensors(tmp_path.string());
    auto params_b = injector_b.get_trainable_params();

    const size_t expected_params = static_cast<size_t>(cfg.num_hidden_layers) * 7 * 2;
    bool count_ok =
        params_a.size() == expected_params &&
        params_b.size() == expected_params &&
        params_a.size() == params_b.size();

    if (!count_ok) {
        std::cerr << "Gemma LoRA param count mismatch: expected " << expected_params
                  << ", got " << params_a.size() << " and " << params_b.size() << std::endl;
        std::filesystem::remove(tmp_path);
        return 1;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < params_a.size(); ++i) {
        max_diff = std::max(max_diff, rel_diff(params_a[i], params_b[i]));
    }

    std::filesystem::remove(tmp_path);

    std::cout << "[GemmaLoRARoundTrip] params=" << params_a.size()
              << " max_diff=" << max_diff << std::endl;

    bool ok = max_diff < 1e-6f;
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}
