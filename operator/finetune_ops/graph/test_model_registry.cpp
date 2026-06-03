#include "model_registry.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

fs::path make_case_dir(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("mft_model_registry_" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

void test_qwen_registry() {
    const fs::path root = make_case_dir("qwen");
    write_file(root / "config.json",
               R"({"model_type":"qwen2","tie_word_embeddings":false})");
    write_file(root / "tokenizer.json", "{}");
    write_file(root / "vocab.json", "{}");
    write_file(root / "merges.txt", "#version: 0.2\n");
    write_file(root / "model.safetensors.index.json", "{}");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Qwen, "qwen family inference failed");
    require(spec.model_type == "qwen2", "qwen model_type mismatch");
    require(!spec.tie_word_embeddings, "qwen tie_word_embeddings should be false");
    require(spec.assets.has_tokenizer_json, "qwen tokenizer.json not detected");
    require(spec.assets.has_vocab_json, "qwen vocab.json not detected");
    require(spec.assets.has_merges_txt, "qwen merges.txt not detected");
    require(!spec.assets.has_single_safetensors, "qwen single safetensors should be absent");
    require(spec.assets.has_sharded_safetensors, "qwen safetensors index not detected");
    require(spec.default_lora_targets.size() == 4, "qwen default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "qwen first default LoRA target mismatch");
    require(ops::to_string(spec.family) == "qwen", "qwen family string mismatch");
}

void test_gpt2_registry_defaults() {
    const fs::path root = make_case_dir("gpt2");
    write_file(root / "config.json", R"({"model_type":"gpt2"})");
    write_file(root / "model.safetensors", "");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::GPT2, "gpt2 family inference failed");
    require(spec.tie_word_embeddings, "gpt2 tie_word_embeddings should default true");
    require(spec.assets.has_single_safetensors, "gpt2 single safetensors not detected");
    require(!spec.assets.has_sharded_safetensors, "gpt2 sharded index should be absent");
    require(spec.default_lora_targets.size() == 2, "gpt2 default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "attn_qkv", "gpt2 first LoRA target mismatch");
}

void test_gemma_registry_defaults_match_training_path() {
    const fs::path root = make_case_dir("gemma");
    write_file(root / "config.json", R"({"model_type":"gemma3_text"})");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Gemma, "gemma family inference failed");
    require(spec.default_lora_targets.size() == 7, "gemma default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "gemma first LoRA target mismatch");
    require(spec.default_lora_targets[4] == "gate_proj", "gemma MLP gate target missing");
    require(spec.default_lora_targets[6] == "down_proj", "gemma MLP down target missing");
}

void test_unknown_model_type() {
    const fs::path root = make_case_dir("unknown");
    write_file(root / "config.json", R"({"model_type":"llama"})");

    bool threw = false;
    try {
        (void)ops::ModelRegistry::inspect_pretrained(root.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "unknown model_type did not throw");
}

}  // namespace

int main() {
    test_qwen_registry();
    test_gpt2_registry_defaults();
    test_gemma_registry_defaults_match_training_path();
    test_unknown_model_type();
    return 0;
}
