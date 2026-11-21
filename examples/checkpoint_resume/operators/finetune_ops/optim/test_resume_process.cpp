// Fresh-process worker for tests/test_resume_process.py. Uses the real tiny Qwen
// forward/backward path and TrainingSession; no pretrained assets are required.
#include "../graph/qwen_model.h"
#include "../core/autograd_engine.h"
#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../core/training_session.h"
#include "gradient_accumulator.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace {
using namespace ops;
using namespace ops::resume;
namespace fs = std::filesystem;
constexpr int crash_exit_code = 86;
constexpr float initial_lr = 0.003f;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct Options {
    std::string mode, run_dir, data;
    int total_steps = 0, grad_accum = 0, stop_after = 0, crash_after = 0;
};

Options parse(int argc, char** argv) {
    Options options;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        require(i + 1 < argc && seen.insert(key).second, "Missing/duplicate option: " + key);
        const std::string value = argv[i + 1];
        if (key == "--mode") options.mode = value;
        else if (key == "--run-dir") options.run_dir = fs::absolute(value).lexically_normal().string();
        else if (key == "--data") options.data = fs::canonical(value).string();
        else {
            size_t used = 0;
            const int number = std::stoi(value, &used);
            require(used == value.size() && number >= 0, "Invalid integer: " + key);
            if (key == "--total-steps") options.total_steps = number;
            else if (key == "--grad-accum") options.grad_accum = number;
            else if (key == "--stop-after") options.stop_after = number;
            else if (key == "--crash-after") options.crash_after = number;
            else throw std::runtime_error("Unknown option: " + key);
        }
    }
    require(options.mode == "new" || options.mode == "resume", "--mode must be new or resume");
    require(!options.run_dir.empty() && !options.data.empty(), "--run-dir and --data are required");
    require(options.total_steps > 0 && options.total_steps <= 200 &&
            options.grad_accum > 0 && options.grad_accum <= 8, "Worker bounds: steps 1..200, accumulation 1..8");
    require(options.stop_after <= options.total_steps && options.crash_after <= options.total_steps,
            "Stop/crash must be within the training budget");
    return options;
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t double_bits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float learning_rate(int update, int total) {
    require(update >= 1 && update <= total, "Scheduler update out of range");
    const double fraction = total == 1 ? 0 : double(update - 1) / (total - 1);
    return static_cast<float>(initial_lr * (0.1 + 0.9 * 0.5 * (1 + std::cos(std::acos(-1.0) * fraction))));
}

void initialize(QwenModel& model) {
    // Same deterministic fixture construction as test_qwen_optimizations, scaled
    // down to one block. Overwrite every random LoRA value; dropout is zero.
    const auto& c = model.config();
    auto assign = [&](const std::string& key, std::vector<int64_t> shape, bool norm = false) {
        auto tensor = std::make_shared<Tensor>(shape, kFloat32, kCPU);
        for (int64_t i = 0; i < tensor->numel(); ++i)
            tensor->data<float>()[i] = norm ? 1.0f : 0.07f * std::sin(float(i + key.size()));
        model.assign_weight(key, tensor);
    };
    assign("embed_tokens.weight", {c.vocab_size, c.hidden_size});
    assign("final_norm.weight", {c.hidden_size}, true);
    const int h = c.hidden_size, kv = h / c.num_attention_heads * c.num_key_value_heads;
    for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
        const auto prefix = "layers." + std::to_string(layer) + ".";
        assign(prefix + "input_norm.weight", {h}, true);
        assign(prefix + "post_norm.weight", {h}, true);
        for (const std::string name : {"q", "k", "v", "o"}) {
            const int out = name == "k" || name == "v" ? kv : h;
            assign(prefix + "self_attn." + name + "_proj.weight", {h, out});
            if (name != "o") assign(prefix + "self_attn." + name + "_proj.bias", {out});
        }
        assign(prefix + "mlp.gate_proj.weight", {h, c.intermediate_size});
        assign(prefix + "mlp.up_proj.weight", {h, c.intermediate_size});
        assign(prefix + "mlp.down_proj.weight", {c.intermediate_size, h});
    }
    model.init_lora(2, 4, 0, true);
    model.freeze_base();
    size_t parameter = 0;
    for (const auto& p : model.get_lora_parameters()) {
        for (int64_t j = 0; j < p->numel(); ++j)
            p->data<float>()[j] = 0.01f * std::cos(float(j + parameter));
        ++parameter;
    }
}

NamedParameters named_parameters(QwenModel& model) {
    NamedParameters result;
    for (int i = 0; i < model.config().num_hidden_layers; ++i) {
        const auto& block = model.get_block(i);
        auto add = [&](const char* projection, const std::unique_ptr<LoRALinear>& linear) {
            require(linear && linear->slices().size() == 1, "Expected one LoRA slice");
            const auto prefix = "layers." + std::to_string(i) + ".self_attn." + projection;
            result.emplace_back(prefix + ".lora_A", linear->slices()[0].A);
            result.emplace_back(prefix + ".lora_B", linear->slices()[0].B);
        };
        add("q_proj", block.q_lin);
        add("v_proj", block.v_lin);
    }
    const auto trainable = model.get_lora_parameters();
    require(result.size() == trainable.size(), "Named trainable coverage mismatch");
    for (size_t i = 0; i < result.size(); ++i)
        require(result[i].second == trainable[i], "Trainable order mismatch");
    return result;
}

Json snapshot(TrainingSession& session, const std::string& filename,
              const NamedParameters& parameters, const Adam& optimizer) {
    const auto state = optimizer.export_named_state(parameters);
    TensorMap tensors;
    for (const auto& [name, tensor] : parameters) tensors.emplace("parameter/" + name, tensor);
    for (const auto& [name, tensor] : state.tensors) tensors.emplace("optimizer/" + name, tensor);
    const auto path = session.artifact_path(filename);
    write_tensor_file(path, tensors);
    return {{"path", path}, {"optimizer", state.metadata}};
}

Json batch_record(const Batch& batch) {
    const size_t count = static_cast<size_t>(batch.input_ids->numel());
    const auto ids = batch.input_ids->data<int32_t>();
    const auto labels = batch.labels->data<int32_t>();
    const auto mask = batch.attention_mask->data<float>();
    return {{"shape", batch.input_ids->shape()}, {"ids", std::vector<int32_t>(ids, ids + count)},
            {"labels", std::vector<int32_t>(labels, labels + count)},
            {"attention_mask", std::vector<float>(mask, mask + count)}};
}

int run(int argc, char** argv) {
    const auto options = parse(argc, argv);
    // Execution controls are deliberately not training arguments. Total budget,
    // data identity, model version and accumulation remain fixed across sessions.
    const std::vector<std::string> arguments{fs::canonical(argv[0]).string(),
        "--data", options.data, "--total-steps", std::to_string(options.total_steps),
        "--grad-accum", std::to_string(options.grad_accum), "--fixture", "tiny-qwen-v1"};
    TrainingSession session(options.run_dir, options.mode == "resume", "test_resume_process_v1",
                            arguments, 1, options.stop_after, 0);
    session.track_assets({options.data});
    QwenConfig model_config;
    model_config.vocab_size = 32; model_config.hidden_size = 8; model_config.intermediate_size = 16;
    model_config.num_attention_heads = 2; model_config.num_key_value_heads = 1;
    model_config.num_hidden_layers = 1;
    QwenModel model(model_config);
    initialize(model);
    const auto named = named_parameters(model);
    const auto parameters = model.get_lora_parameters();
    AdamW optimizer(AdamWConfig(initial_lr, 0.9f, 0.99f, 1e-8f, 0.01f, 0.0f, true));
    WT2Config data_config;
    data_config.seq_len = 8; data_config.jsonl_train = options.data;
    data_config.shuffle_train = true; data_config.drop_last = false; data_config.seed = 147;
    WikiText2Dataset dataset(data_config, static_cast<HFTokenizer*>(nullptr));
    dataset.load(Split::Train);
    Json trainer{{"completed_updates", 0}, {"total_steps", options.total_steps},
        {"grad_accum", options.grad_accum}, {"micro_batch_size", 2},
        {"scheduler", "cosine_v1"}, {"initial_lr", initial_lr}, {"minimum_lr_ratio", 0.1},
        {"lr_bits", float_bits(initial_lr)}, {"tokens_seen", 0}, {"targets_seen", 0},
        {"best_path", nullptr}, {"best_update", 0}, {"best_score", nullptr},
        {"best_policy", "synthetic_negative_update_for_pointer_test"}};
    if (auto restored = session.restore(named, optimizer)) {
        require(!restored->fallback, "Unexpected fallback in process equivalence test");
        trainer = restored->trainer;
        require(trainer.at("total_steps") == options.total_steps && trainer.at("grad_accum") == options.grad_accum,
                "Restored training configuration mismatch");
        dataset.import_state(restored->dataset_state);
        require(trainer.at("lr_bits") == float_bits(optimizer.get_learning_rate()), "Current LR was not restored");
    }
    const int start = trainer.at("completed_updates").get<int>();
    require(!options.crash_after || options.crash_after > start, "Crash point has already passed");
    session.event({{"type", "worker_start"}, {"pid", ::getpid()}, {"mode", options.mode},
                   {"model", "tiny_qwen_1x8_lora_qv_rank2"}, {"completed_updates", start},
                   {"trainer", trainer}, {"dataset", Json::parse(dataset.export_state())},
                   {"snapshot", snapshot(session, "start.safetensors", named, optimizer)}});
    if (options.mode == "new") session.commit(named, optimizer, dataset.export_state(), trainer, "running");
    for (int update = start + 1; update <= options.total_steps; ++update) {
        const auto before = Json::parse(dataset.export_state());
        std::vector<Batch> microbatches;
        int64_t total_targets = 0, total_tokens = 0;
        for (int micro = 0; micro < options.grad_accum; ++micro) {
            // Own the reusable dataset buffers before prefetching another batch.
            microbatches.push_back(own_microbatch(dataset.next_batch(2, true)));
            const auto& batch = microbatches.back();
            total_targets += valid_lm_label_count(batch.labels, model_config.vocab_size);
            for (int64_t j = 0; j < batch.attention_mask->numel(); ++j)
                total_tokens += batch.attention_mask->data<float>()[j] != 0;
        }
        require(total_targets > 0, "No supervised tokens");
        for (const auto& parameter : parameters) parameter->zero_grad();
        GradientAccumulator accumulator(parameters.size());
        Json micro_trace = Json::array();
        double loss_sum = 0;
        for (const auto& batch : microbatches) {
            const auto targets = valid_lm_label_count(batch.labels, model_config.vocab_size);
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            const float value = loss->data<float>()[0];
            require(std::isfinite(value), "Nonfinite model loss");
            loss_sum += double(value) * targets / total_targets;
            mul(loss, float(targets) / total_targets)->backward();
            accumulator.collect(parameters);
            Json trace = batch_record(batch);
            trace["loss_bits"] = float_bits(value);
            trace["valid_targets"] = targets;
            micro_trace.push_back(std::move(trace));
        }
        accumulator.assign(parameters);
        std::vector<TensorPtr> gradients;
        double squared_norm = 0;
        for (const auto& p : parameters) {
            require(p->grad() != nullptr, "Missing real LoRA gradient");
            gradients.push_back(p->grad());
            for (int64_t i = 0; i < p->numel(); ++i) {
                const double g = p->grad()->data<float>()[i];
                require(std::isfinite(g), "Nonfinite gradient");
                squared_norm += g * g;
            }
        }
        require(squared_norm > 0, "No real autograd gradient");
        optimizer.set_learning_rate(learning_rate(update, options.total_steps));
        optimizer.step(parameters, gradients);
        for (const auto& p : parameters) p->zero_grad();
        autograd::Engine::instance().clear_graph();
        trainer["completed_updates"] = update;
        trainer["tokens_seen"] = trainer.at("tokens_seen").get<int64_t>() + total_tokens;
        trainer["targets_seen"] = trainer.at("targets_seen").get<int64_t>() + total_targets;
        trainer["lr_bits"] = float_bits(optimizer.get_learning_rate());
        trainer["loss_bits"] = double_bits(loss_sum);
        // A controlled, explicitly synthetic best score guarantees that the
        // crashing update creates a newer uncommitted candidate on every run.
        TensorMap adapter;
        for (const auto& [name, tensor] : named) adapter.emplace(name, tensor);
        const auto best_path = session.artifact_path("best-" + std::to_string(update) + ".safetensors");
        write_tensor_file(best_path, adapter);
        trainer["best_path"] = best_path;
        trainer["best_update"] = update;
        trainer["best_score"] = -update;
        auto full = snapshot(session, "update-" + std::to_string(update) + ".safetensors", named, optimizer);
        session.event({{"type", "update"}, {"completed_updates", update}, {"pid", ::getpid()},
            {"loss", loss_sum}, {"loss_bits", double_bits(loss_sum)},
            {"lr", optimizer.get_learning_rate()}, {"lr_bits", trainer.at("lr_bits")},
            {"gradient_squared_norm_bits", double_bits(squared_norm)},
            {"tokens", total_tokens}, {"targets", total_targets}, {"microbatches", micro_trace},
            {"dataset_before", before}, {"dataset_after", Json::parse(dataset.export_state())},
            {"trainer", trainer}, {"snapshot", full}});
        if (update == options.crash_after) {
            // Process termination, not a caught exception or an in-process reset.
            // TrainingSession::event uses write(2), so the attempt is visible even
            // though no checkpoint commit/destructor is allowed to run.
            ::_exit(crash_exit_code);
        }
        const bool suspended = session.suspend_requested(update) && update < options.total_steps;
        const std::string status = update == options.total_steps ? "completed" : suspended ? "suspended" : "running";
        session.commit(named, optimizer, dataset.export_state(), trainer, status);
        if (suspended) return 0;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "resume process worker failed: " << error.what() << '\n';
        return 1;
    }
}
