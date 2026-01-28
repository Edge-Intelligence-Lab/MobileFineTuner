#pragma once

#include "gemma_model.h"
#include <string>
#include <vector>

namespace ops {

struct GemmaLoraSpec {
    int rank = 8;
    float alpha = 32.0f;
    float dropout = 0.1f;
    std::vector<std::string> target_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    };
    std::vector<int> layers;  // empty = all layers

    static GemmaLoraSpec full_attn_mlp() {
        return GemmaLoraSpec();
    }

    static GemmaLoraSpec attention_only() {
        GemmaLoraSpec spec;
        spec.target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"};
        return spec;
    }

    static GemmaLoraSpec attention_light() {
        GemmaLoraSpec spec;
        spec.target_modules = {"q_proj", "v_proj"};
        return spec;
    }
};

class GemmaLoraInjector {
public:
    GemmaLoraInjector() = default;
    ~GemmaLoraInjector() = default;

    void inject(GemmaModel& model, const GemmaLoraSpec& spec);
    std::vector<TensorPtr> get_trainable_params() const;
    void print_info() const;

    void save_lora_safetensors(const std::string& path) const;
    void load_lora_safetensors(const std::string& path);

    void merge_all(GemmaModel& model);
    void unmerge_all(GemmaModel& model);

private:
    GemmaModel* model_ = nullptr;
    GemmaLoraSpec spec_;
    int num_layers_ = 0;
    int attached_modules_ = 0;
};

}  // namespace ops
