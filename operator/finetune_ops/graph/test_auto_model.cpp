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
    bool duplicate_rejected = false;
    try {
        model->init_lora(tiny_lora());
    } catch (const std::exception&) {
        duplicate_rejected = true;
    }
    require(duplicate_rejected,
            "Repeated LoRA initialization must reject stale trainer references");
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
    auto cfg = tiny_lora();
    cfg.dropout = 0.2f;
    model->init_lora(cfg);
    require(model->gemma()->get_block(0).q_proj_lora->dropout() == cfg.dropout,
            "Gemma ignored configured LoRA dropout");
    bool rejected = false;
    try {
        model->gemma()->set_activation_checkpointing(true);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Gemma checkpointing must reject stochastic LoRA recomputation");
    auto checkpointed = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    checkpointed->gemma()->set_activation_checkpointing(true);
    rejected = false;
    try {
        checkpointed->init_lora(cfg);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Gemma accepted dropout after enabling checkpointing");
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

void test_qwen3_exact_target_dispatch() {
    const auto root = make_case_dir("qwen3");
    write_file(
        root / "config.json",
        R"({"model_type":"qwen3","vocab_size":32,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":6,"max_position_embeddings":16})");
    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    auto lora = tiny_lora();
    lora.target_modules = {"k_proj"};
    model->init_lora(lora);
    auto params = model->trainable_parameters();
    require(params.size() == 2, "Auto API ignored exact Qwen3 LoRA targets");
    require(params[0]->shape() == std::vector<int64_t>({8, 2}), "Qwen3 LoRA input dimensions");
    require(params[1]->shape() == std::vector<int64_t>({2, 6}), "Qwen3 LoRA output dimensions");
}

#ifdef USE_NEW_AUTOGRAD_ENGINE
void test_qwen35_auto_trainer() {
    const auto root = make_case_dir("qwen35");
    write_file(
        root / "config.json",
        R"({"model_type":"qwen3_5_text","vocab_size":32,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"tie_word_embeddings":true,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":3,"linear_value_head_dim":4,"linear_conv_kernel_dim":3,"layer_types":["linear_attention","full_attention"],"rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5,"rope_theta":10000}})");
    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    for (auto& p : model->parameters())
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = p->ndim() == 1 ? 0.5f : 0.05f * std::sin(float(i + 1));
    model->init_lora(tiny_lora());
    require(model->trainable_parameters().size() == 18, "Hybrid Auto API default target count");
    require(model->spec().default_lora_targets == model->qwen()->lora_targets(),
            "Hybrid Auto API default metadata mismatch");
    auto ids = std::make_shared<ops::Tensor>(std::vector<int64_t>{1, 4}, ops::kInt32, ops::kCPU);
    for (int i = 0; i < 4; ++i)
        ids->data<int32_t>()[i] = i + 1;
    auto params = model->trainable_parameters();
    std::vector<std::vector<float>> original;
    for (const auto& p : params)
        original.emplace_back(p->data<float>(), p->data<float>() + p->numel());
    ops::AutoTrainer trainer(*model);
    auto wrong_labels =
        std::make_shared<ops::Tensor>(std::vector<int64_t>{1, 3}, ops::kInt32, ops::kCPU);
    bool rejected = false;
    try {
        trainer.train_step(ids, nullptr, wrong_labels);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Hybrid trainer accepted mismatched labels");
    for (const auto& p : params)
        if (p->grad())
            for (int64_t i = 0; i < p->grad()->numel(); ++i)
                require(p->grad()->data<float>()[i] == 0, "Failed training left stale gradients");
    for (int step = 0; step < 2; ++step) {
        const auto result = trainer.train_step(ids, nullptr, ids);
        require(std::isfinite(result.loss), "Hybrid AutoTrainer non-finite loss");
    }
    double change = 0;
    for (size_t p = 0; p < params.size(); ++p)
        for (int64_t i = 0; i < params[p]->numel(); ++i)
            change += std::abs(params[p]->data<float>()[i] - original[p][i]);
    require(change > 0, "Hybrid AutoTrainer did not update adapter weights");
}
#endif

}  // namespace

int main() {
    test_gpt2_auto_trainer_step();
    test_gemma_auto_lora_dispatch();
    test_qwen_auto_lora_dispatch();
    test_qwen3_exact_target_dispatch();
#ifdef USE_NEW_AUTOGRAD_ENGINE
    test_qwen35_auto_trainer();
#endif
    return 0;
}
