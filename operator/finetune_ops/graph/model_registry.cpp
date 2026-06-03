#include "model_registry.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

namespace fs = std::filesystem;

bool path_exists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

std::string read_file(const fs::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string extract_json_string(const std::string& content, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str();
    }
    return "";
}

bool extract_json_bool(const std::string& content, const std::string& key, bool fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str() == "true";
    }
    return fallback;
}

ops::ModelFamily family_from_model_type(const std::string& model_type) {
    const std::string normalized = lowercase(model_type);
    if (normalized.find("qwen") != std::string::npos) {
        return ops::ModelFamily::Qwen;
    }
    if (normalized.find("gemma") != std::string::npos) {
        return ops::ModelFamily::Gemma;
    }
    if (normalized.find("gpt2") != std::string::npos ||
        normalized.find("gpt_2") != std::string::npos ||
        normalized == "gpt") {
        return ops::ModelFamily::GPT2;
    }
    return ops::ModelFamily::Unknown;
}

std::vector<std::string> default_lora_targets(ops::ModelFamily family) {
    switch (family) {
        case ops::ModelFamily::Qwen:
            return {"q_proj", "k_proj", "v_proj", "o_proj"};
        case ops::ModelFamily::Gemma:
            return {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"};
        case ops::ModelFamily::GPT2:
            return {"attn_qkv", "attn_proj"};
        case ops::ModelFamily::Unknown:
            return {};
    }
    return {};
}

}  // namespace

namespace ops {

std::string to_string(ModelFamily family) {
    switch (family) {
        case ModelFamily::GPT2: return "gpt2";
        case ModelFamily::Gemma: return "gemma";
        case ModelFamily::Qwen: return "qwen";
        case ModelFamily::Unknown: return "unknown";
    }
    return "unknown";
}

ModelFamily ModelRegistry::infer_family(const std::string& model_dir) {
    return inspect_pretrained(model_dir).family;
}

ModelArchitectureSpec ModelRegistry::inspect_pretrained(const std::string& model_dir) {
    const fs::path root(model_dir);
    const fs::path config_path = root / "config.json";
    if (!path_exists(config_path)) {
        throw std::runtime_error("Model directory is missing config.json: " + model_dir);
    }

    const std::string config = read_file(config_path);
    const std::string model_type = extract_json_string(config, "model_type");
    const ModelFamily family = family_from_model_type(model_type);
    if (family == ModelFamily::Unknown) {
        throw std::runtime_error("Unsupported model_type in config.json: " + model_type);
    }

    ModelArchitectureSpec spec;
    spec.family = family;
    spec.model_type = model_type;
    spec.tie_word_embeddings = extract_json_bool(
        config,
        "tie_word_embeddings",
        family == ModelFamily::GPT2);
    spec.default_lora_targets = default_lora_targets(family);

    auto& assets = spec.assets;
    assets.model_dir = model_dir;
    assets.config_json_path = config_path.string();
    assets.tokenizer_json_path = (root / "tokenizer.json").string();
    assets.vocab_json_path = (root / "vocab.json").string();
    assets.merges_txt_path = (root / "merges.txt").string();
    assets.safetensors_path = (root / "model.safetensors").string();
    assets.safetensors_index_path = (root / "model.safetensors.index.json").string();

    assets.has_tokenizer_json = path_exists(assets.tokenizer_json_path);
    assets.has_vocab_json = path_exists(assets.vocab_json_path);
    assets.has_merges_txt = path_exists(assets.merges_txt_path);
    assets.has_single_safetensors = path_exists(assets.safetensors_path);
    assets.has_sharded_safetensors = path_exists(assets.safetensors_index_path);

    return spec;
}

}  // namespace ops
