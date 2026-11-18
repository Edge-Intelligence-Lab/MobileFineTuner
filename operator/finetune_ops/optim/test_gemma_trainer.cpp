#include "../core/autograd_engine.h"
#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../data/wikitext2_dataset.h"
#include "gemma_trainer.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ops;
namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
GemmaTextConfig tiny_config() {
    GemmaTextConfig c;
    c.vocab_size = 32;
    c.hidden_size = 8;
    c.intermediate_size = 16;
    c.num_hidden_layers = 1;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.head_dim = 4;
    c.max_position_embeddings = 16;
    c.sliding_window = 8;
    c.layer_types = {"sliding_attention"};
    return c;
}
void initialize(GemmaModel& model, GemmaLoraInjector& injector) {
    for (const auto& p : model.parameters()) {
        p->set_requires_grad(false);
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = p->ndim() == 1 ? 0.1f : 0.05f * std::sin(float(i + 1));
    }
    auto spec = GemmaLoraSpec::full_attn_mlp();
    spec.rank = 2;
    spec.alpha = 4;
    spec.dropout = 0;
    injector.inject(model, spec);
    for (const auto& p : injector.get_trainable_params())
        for (int64_t i = 0; i < p->numel(); ++i)
            p->data<float>()[i] = 0.02f * std::cos(float(i + 1));
}
void test_tail(const std::filesystem::path& path, float max_norm) {
    WT2Config data;
    data.jsonl_train = path.string();
    data.seq_len = 4;
    data.shuffle_train = false;
    auto encoder = [](const std::string&) { return std::vector<int32_t>{}; };
    WikiText2Dataset actual_data(data, encoder), reference_data(data, encoder);
    actual_data.load(Split::Train);
    reference_data.load(Split::Train);
    require(actual_data.num_sequences() == 3,
            "Fixture must have an incomplete final accumulation group");
    GemmaModel actual(tiny_config()), reference(tiny_config());
    GemmaLoraInjector actual_lora, reference_lora;
    initialize(actual, actual_lora);
    initialize(reference, reference_lora);
    GemmaTrainerConfig config;
    config.learning_rate = 0.001f;
    config.num_epochs = 1;
    config.grad_accum_steps = 2;
    config.micro_batch_size = 1;
    config.lr_scheduler = "constant";
    config.warmup_ratio = 0;
    config.max_grad_norm = max_norm;
    config.eval_steps = 0;
    config.adam_eps = 1e-6f;
    config.logging_steps = 1;
    GemmaLoRATrainer trainer(actual, actual_lora, actual_data, actual_data, config);
    trainer.train();
    require(trainer.global_step() == 2, "Gemma dropped its final short accumulation group");
    AdamConfig adam_config;
    adam_config.learning_rate = config.learning_rate;
    adam_config.epsilon = config.adam_eps;
    Adam optimizer(adam_config);
    auto params = reference_lora.get_trainable_params();
    // Independent reference: mean the two full-group losses, then use a full
    // unscaled loss for the single-sample group. Each group gets one Adam step.
    for (int group_size : {2, 1}) {
        optimizer.zero_grad(params);
        for (int micro = 0; micro < group_size; ++micro) {
            auto batch = reference_data.next_batch(1, false);
            auto loss = lm_cross_entropy(reference.forward(batch.input_ids, batch.attention_mask),
                                         batch.labels);
            mul(loss, 1.0f / group_size)->backward();
        }
        if (max_norm > 0)
            optimizer.clip_grad_norm(params, max_norm);
        std::vector<TensorPtr> grads;
        for (auto& p : params)
            grads.push_back(p->grad());
        optimizer.step(params, grads);
    }
    auto actual_params = actual_lora.get_trainable_params();
    for (size_t p = 0; p < params.size(); ++p)
        for (int64_t i = 0; i < params[p]->numel(); ++i)
            require(std::abs(actual_params[p]->data<float>()[i] - params[p]->data<float>()[i]) <
                        2e-6f,
                    "Gemma tail normalization or shared clipping changed the reference update");
    actual_data.reset_cursor();
    auto wrong = actual_data.next_batch(1, false);
    wrong.labels = zeros({1, 3}, kInt32);
    bool rejected = false;
    try {
        trainer.train_step(wrong);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected && trainer.global_step() == 2, "Failed Gemma step advanced the optimizer");
    for (auto& p : actual_params)
        if (p->grad())
            for (int64_t i = 0; i < p->grad()->numel(); ++i)
                require(p->grad()->data<float>()[i] == 0, "Failed Gemma step left stale gradients");
    actual_data.reset_cursor();
    auto valid = actual_data.next_batch(1, false);
    require(trainer.train_step(valid) < 0, "Gemma failed-step recovery kept partial accumulation");
    require(std::isfinite(trainer.train_step(valid)) && trainer.global_step() == 3,
            "Gemma cannot recover after invalid labels");
    // Dropout-enabled adapters must evaluate deterministically without retaining a graph.
    actual.get_block(0).q_proj_lora->set_dropout(0.5f, 42);
    const auto first_eval = trainer.evaluate();
    const auto second_eval = trainer.evaluate();
    require(first_eval == second_eval && autograd::Engine::instance().is_enabled(),
            "Evaluation left stochastic dropout enabled or failed to restore grad mode");
    auto state = autograd::Engine::instance().take_state();
    require(state.nodes.empty(), "Evaluation retained an autograd graph");
    autograd::Engine::instance().restore_state(std::move(state));
    auto invalid = config;
    invalid.grad_accum_steps = 0;
    rejected = false;
    try {
        GemmaLoRATrainer bad(actual, actual_lora, actual_data, actual_data, invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Gemma accepted zero accumulation span");
}
} // namespace
int main() try {
    auto directory = std::filesystem::temp_directory_path() / "mft_gemma_trainer_tail";
    std::filesystem::create_directories(directory);
    auto path = directory / "samples.jsonl";
    {
        std::ofstream file(path);
        for (int i = 0; i < 3; ++i)
            file << "{\"ids\":[1,2,3,4],\"mask\":[0,1,1,1]}\n";
    }
    test_tail(path, 0);
    test_tail(path, 0.2f);
    std::filesystem::remove_all(directory);
    std::cout << "PASS: Gemma actual-size tail accumulation, shared clipping, finite-step cleanup "
                 "and recovery\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
