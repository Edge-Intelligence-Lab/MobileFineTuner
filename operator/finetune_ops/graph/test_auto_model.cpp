#include "auto_model.h"
#include "../optim/auto_trainer.h"

#include <cmath>
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
    const fs::path root = fs::temp_directory_path() / ("mft_auto_model_" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

ops::AutoModelLoadOptions no_weight_load() {
    ops::AutoModelLoadOptions options;
    options.load_weights = false;
    options.verbose = false;
    return options;
}

ops::AutoLoraConfig tiny_lora() {
    ops::AutoLoraConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 4.0f;
    cfg.dropout = 0.0f;
    cfg.seed = 7;
    return cfg;
}

void fill_parameters(ops::AutoModelForCausalLM& model, float value) {
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = value;
        }
    }
}

void test_gpt2_auto_trainer_step() {
    const fs::path root = make_case_dir("gpt2");
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::GPT2, "GPT-2 family dispatch failed");
    fill_parameters(*model, 0.01f);

    model->init_lora(tiny_lora());
    require(!model->trainable_parameters().empty(), "GPT-2 LoRA parameters missing");

    int32_t ids[] = {1, 2, 3};
    float mask[] = {1.0f, 1.0f, 1.0f};
    int32_t labels[] = {-100, 2, 3};
    auto input_ids = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, ids, ops::kInt32, ops::kCPU);
    auto attention_mask = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, mask, ops::kFloat32, ops::kCPU);
    auto label_tensor = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, labels, ops::kInt32, ops::kCPU);

    ops::AutoTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-3f;
    ops::AutoTrainer trainer(*model, trainer_cfg);
    const auto result = trainer.train_step(input_ids, attention_mask, label_tensor);
    require(std::isfinite(result.loss), "GPT-2 AutoTrainer loss is not finite");
    require(result.trainable_tensor_count > 0, "GPT-2 trainable tensor count missing");
}

void test_gemma_auto_lora_dispatch() {
    const fs::path root = make_case_dir("gemma");
    write_file(root / "config.json",
               R"({"model_type":"gemma3_text","vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"sliding_window":8})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::Gemma, "Gemma family dispatch failed");
    model->init_lora(tiny_lora());
    require(!model->trainable_parameters().empty(), "Gemma LoRA parameters missing");
}

void test_qwen_auto_lora_dispatch() {
    const fs::path root = make_case_dir("qwen");
    write_file(root / "config.json",
               R"({"model_type":"qwen2","vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::Qwen, "Qwen family dispatch failed");
    model->init_lora(ops::AutoLoraConfig::attention_qkvo());
    require(!model->trainable_parameters().empty(), "Qwen LoRA parameters missing");
}

}  // namespace

int main() {
    test_gpt2_auto_trainer_step();
    test_gemma_auto_lora_dispatch();
    test_qwen_auto_lora_dispatch();
    return 0;
}
