#pragma once

#include <string>
#include <vector>

namespace ops {

enum class ModelFamily {
    GPT2,
    Gemma,
    Qwen,
    Unknown,
};

std::string to_string(ModelFamily family);

// C++ graph capabilities, not CLI flags or a device-memory guarantee.
// Availability is reported for the engine variant in this library build.
struct ModelTrainingCapabilities {
    bool causal_lm_lora = false;
    bool streamed_attention = false;
    bool activation_checkpointing = false;
    bool frozen_parameter_offload = false;
    bool standard_peft_adapter_io = false;
};

struct ModelAssetPaths {
    std::string model_dir;
    std::string config_json_path;
    std::string tokenizer_json_path;
    std::string vocab_json_path;
    std::string merges_txt_path;
    std::string safetensors_path;
    std::string safetensors_index_path;

    bool has_tokenizer_json = false;
    bool has_vocab_json = false;
    bool has_merges_txt = false;
    bool has_single_safetensors = false;
    bool has_sharded_safetensors = false;
};

struct ModelArchitectureSpec {
    ModelFamily family = ModelFamily::Unknown;
    std::string model_type;
    ModelAssetPaths assets;
    std::vector<std::string> default_lora_targets;
    bool tie_word_embeddings = false;
    ModelTrainingCapabilities capabilities{};
};

class ModelRegistry {
public:
    static ModelArchitectureSpec inspect_pretrained(const std::string& model_dir);
    static ModelFamily infer_family(const std::string& model_dir);
    static ModelTrainingCapabilities capabilities_for(const std::string& model_type);
};

}  // namespace ops
