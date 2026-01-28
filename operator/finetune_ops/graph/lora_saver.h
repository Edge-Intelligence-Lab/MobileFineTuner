/**
 * @file lora_saver.h
 * @brief LoRA safetensors save/load (PEFT compatible)
 */

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "../core/tensor.h"
#include "gpt2_model.h"
#include "lora_injector.h"

namespace ops {

struct LoRAState {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;
    bool split_qkv = true;
    std::vector<LoraTarget> targets;
    
    // A/B tensors per layer/target
    // key: "layer.{i}.{target}.lora_A" / "lora_B"
    std::unordered_map<std::string, TensorPtr> tensors;
    
    bool compatible_with(const GPT2Model& model) const;
};

class LoraSaver {
public:
    // Save: model must already have LoRA injected
    static void save_safetensors(const std::string& path, 
                                  const GPT2Model& model,
                                  const LoraSpec& spec);
    
    // Load: return state for external attachment to model
    static LoRAState load_safetensors(const std::string& path);
    
    // Attach loaded state to model
    static void attach_from_state(GPT2Model& model, const LoRAState& state);
    
    // Helper methods (public for external use)
    static std::string make_peft_key(int layer, const std::string& target, const std::string& ab);
    static bool parse_peft_key(const std::string& key, int& layer, std::string& target, std::string& ab);
};

} // namespace ops
