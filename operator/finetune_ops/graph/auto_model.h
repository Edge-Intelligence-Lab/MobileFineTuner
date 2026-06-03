#pragma once

#include "../core/tensor.h"
#include "gemma_lora_injector.h"
#include "gemma_model.h"
#include "gpt2_model.h"
#include "lora_injector.h"
#include "model_registry.h"
#include "qwen_model.h"
#include "safetensors_loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ops {

struct AutoModelLoadOptions {
    bool load_weights = true;
    bool verbose = false;
    SafeTensorsLoadOptions safetensors_options;

    AutoModelLoadOptions() {
        safetensors_options.verbose = false;
    }
};

struct AutoLoraConfig {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.05f;
    uint64_t seed = 42;
    std::vector<std::string> target_modules;

    static AutoLoraConfig attention_qkvo() {
        AutoLoraConfig cfg;
        cfg.target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"};
        return cfg;
    }
};

class AutoModelForCausalLM {
public:
    static std::unique_ptr<AutoModelForCausalLM>
    from_pretrained(const std::string& model_dir,
                    const AutoModelLoadOptions& options = AutoModelLoadOptions());

    TensorPtr forward(const TensorPtr& input_ids,
                      const TensorPtr& attention_mask = nullptr);

    void init_lora(const AutoLoraConfig& config = AutoLoraConfig());

    std::vector<TensorPtr> parameters();
    std::vector<TensorPtr> trainable_parameters();

    ModelFamily family() const { return spec_.family; }
    const ModelArchitectureSpec& spec() const { return spec_; }
    const std::string& model_dir() const { return model_dir_; }

    GPT2Model* gpt2() { return gpt2_.get(); }
    GemmaModel* gemma() { return gemma_.get(); }
    QwenModel* qwen() { return qwen_.get(); }

private:
    AutoModelForCausalLM(std::string model_dir, ModelArchitectureSpec spec);

    void load_pretrained_weights(const AutoModelLoadOptions& options);

    std::string model_dir_;
    ModelArchitectureSpec spec_;

    std::unique_ptr<GPT2Model> gpt2_;
    std::unique_ptr<GemmaModel> gemma_;
    std::unique_ptr<QwenModel> qwen_;

    std::unique_ptr<LoraInjector> gpt2_lora_;
    std::unique_ptr<GemmaLoraInjector> gemma_lora_;
};

}  // namespace ops
