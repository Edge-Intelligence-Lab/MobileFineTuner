#pragma once

#include "finetune_ops/core/training_session.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/nn/lora_linear.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace resume_cli {
using ops::resume::Json;

inline void validate_physical_batch(int logical, int physical) {
    if (physical < 1 || physical > logical)
        throw std::runtime_error("Physical microbatch must be between 1 and the original --batch");
}

inline int64_t physical_accumulation(int logical, int physical, int accumulation) {
    // Logical loader batches are split independently; never merge their tails.
    return static_cast<int64_t>(accumulation) * (1 + (logical - 1) / physical);
}

template <class Parser>
auto batch_validator(Parser parse) {
    return [parse](std::vector<std::string>& args, int physical) {
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        validate_physical_batch(parse(static_cast<int>(argv.size()), argv.data()).batch, physical);
    };
}

template <class Options>
void configure(Options& cli, ops::resume::TrainingSession& session) {
    if (cli.batch > std::numeric_limits<int>::max() / cli.grad_accum)
        throw std::runtime_error("Effective batch is too large");
    if (std::getenv("QWEN_DEBUG") || std::getenv("DISABLE_QK_NORM")) {
        throw std::runtime_error("Session mode rejects QWEN_DEBUG and DISABLE_QK_NORM computation overrides");
    }
    if (cli.lora_dropout != 0 || cli.shard_fp16_disk || cli.dump_first_batch ||
        !cli.dump_first_batch_path.empty() || cli.exit_after_dump ||
        cli.dump_first_tokens != 16 || !cli.fixed_batch_bin.empty() ||
        cli.fixed_batch_steps != 0 || cli.stop_after_steps != 0) {
        throw std::runtime_error("Session mode does not support dropout, FP16 shards, debug dumps, "
                                 "fixed batches or stop_after_steps; use suspend_after_updates");
    }
    if (cli.target_mode != "qv" && cli.target_mode != "full") {
        throw std::runtime_error("Session mode requires target_mode=qv or full");
    }
    // prepare_invocation restores the original working directory before parsing.
    auto canonical = [](std::string& path) {
        if (!path.empty()) path = std::filesystem::canonical(path).string();
    };
    canonical(cli.model_dir);
    std::vector<std::string> assets{cli.model_dir + "/config.json",
                                    cli.model_dir + "/model.safetensors"};
    if (!cli.pretokenized_path.empty()) {
        if (cli.pretokenized_meta.empty()) {
            const std::filesystem::path bin(cli.pretokenized_path);
            auto replaced = bin;
            replaced.replace_extension(".json");
            // Match WikiText2Dataset's metadata discovery order exactly.
            for (const auto& candidate : std::vector<std::string>{
                     (bin.parent_path() / "meta.json").string(), bin.string() + ".json",
                     bin.string() + ".meta.json", replaced.string()}) {
                std::ifstream in(candidate);
                if (in.good()) { cli.pretokenized_meta = candidate; break; }
            }
            if (cli.pretokenized_meta.empty()) throw std::runtime_error("Cannot locate pretokenized metadata");
        }
        canonical(cli.pretokenized_path);
        canonical(cli.pretokenized_meta);
        assets.push_back(cli.pretokenized_path);
        assets.push_back(cli.pretokenized_meta);
    } else if (!cli.jsonl_train.empty()) {
        canonical(cli.jsonl_train);
        canonical(cli.jsonl_valid);
        assets.push_back(cli.jsonl_train);
        assets.push_back(cli.jsonl_valid.empty() ? cli.jsonl_train : cli.jsonl_valid);
    } else {
        canonical(cli.data_dir);
        assets.push_back(cli.data_dir + "/wiki.train.raw");
        assets.push_back(cli.data_dir + "/wiki.valid.raw");
        canonical(cli.tokenizer_json);
        assets.push_back(cli.tokenizer_json.empty() ? cli.model_dir + "/tokenizer.json" : cli.tokenizer_json);
    }
    canonical(cli.resume_from);
    if (!cli.resume_from.empty()) assets.push_back(cli.resume_from);
    for (auto& file : assets) canonical(file);
    std::sort(assets.begin(), assets.end());
    assets.erase(std::unique(assets.begin(), assets.end()), assets.end());
    session.track_assets(assets);
    cli.output_dir = std::filesystem::absolute(session.artifact_path("lora.safetensors"))
                         .parent_path().lexically_normal().string();
    cli.measurement_dir = session.session_dir();
    cli.eval_out = session.session_dir() + "/eval.jsonl";
    // Shards are disposable session-local caches, never checkpoint authority.
    cli.shard_dir = cli.output_dir + "/shards";
}

inline void add_linear(ops::NamedParameters& named, Json& layout, int layer,
                       const std::string& target, const ops::LoRALinear* linear) {
    if (!linear) return;
    const auto& slices = linear->slices();
    // Warm-start loaders do not promise stable ordering for multiple slices.
    if (slices.size() > 1) throw std::runtime_error("Session mode requires one LoRA slice per target");
    for (const auto& slice : slices) {
        if (!slice.A || !slice.B || !std::isfinite(slice.scale)) {
            throw std::runtime_error("Invalid LoRA slice in session mode");
        }
        const auto base = "layer." + std::to_string(layer) + "." + target;
        named.emplace_back(base + ".lora_A", slice.A);
        named.emplace_back(base + ".lora_B", slice.B);
        layout.push_back({{"name", base}, {"scale", slice.scale},
                          {"col0", slice.col0}, {"cols", slice.cols}});
    }
}

inline void validate_parameters(const ops::NamedParameters& named,
                                const std::vector<ops::TensorPtr>& trainable) {
    if (named.empty() || named.size() != trainable.size()) {
        throw std::runtime_error("Named LoRA parameters do not cover all trainable tensors");
    }
    std::set<std::string> names;
    std::set<const ops::Tensor*> pointers;
    for (size_t i = 0; i < named.size(); ++i) {
        if (named[i].second != trainable[i] || !names.insert(named[i].first).second ||
            !pointers.insert(named[i].second.get()).second) {
            throw std::runtime_error("Unsupported LoRA aliases or optimizer parameter order");
        }
    }
}

inline Json finite_or_null(float value) {
    return std::isfinite(value) ? Json(value) : Json(nullptr);
}

struct TrainerState {
    int completed_updates = 0;
    int physical_batch = 0;
    float ema_loss = 0;
    bool ema_init = false;
    int64_t total_tokens = 0;
    float best_valid_ppl = std::numeric_limits<float>::infinity();
    std::string best_path;
    Json timing = {{"updates", 0}, {"core_step_sec", 0.0}, {"full_step_sec", 0.0},
                   {"forward_sec", 0.0}, {"loss_sec", 0.0}, {"backward_sec", 0.0},
                   {"optimizer_sec", 0.0}, {"cleanup_sec", 0.0}, {"eval_sec", 0.0},
                   {"evaluations", 0}};

    Json json(int total_steps, const Json& layout) const {
        return {{"schema_version", 1}, {"completed_updates", completed_updates}, {"physical_batch", physical_batch},
                {"total_steps", total_steps}, {"ema_init", ema_init}, {"ema_loss", ema_loss},
                {"total_tokens", total_tokens}, {"best_valid_ppl", finite_or_null(best_valid_ppl)},
                {"best_path", best_path}, {"timing", timing}, {"adapter_layout", layout}};
    }

    void restore(const Json& j, int total_steps, const Json& layout) {
        if (!j.at("schema_version").is_number_integer() || j.at("schema_version") != 1 ||
            !j.at("total_steps").is_number_integer() || j.at("total_steps") != total_steps ||
            j.at("adapter_layout") != layout) {
            throw std::runtime_error("Unsupported trainer schema, schedule budget or adapter layout");
        }
        auto integer = [](const Json& value) -> int64_t {
            if (!value.is_number_integer() || value.get<double>() < 0 ||
                value.get<double>() >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("Invalid nonnegative trainer counter");
            }
            return value.get<int64_t>();
        };
        const auto completed = integer(j.at("completed_updates"));
        if (completed > total_steps || !j.at("ema_init").is_boolean()) {
            throw std::runtime_error("Invalid trainer progress");
        }
        completed_updates = static_cast<int>(completed);
        total_tokens = integer(j.at("total_tokens"));
        ema_init = j.at("ema_init").get<bool>();
        ema_loss = j.at("ema_loss").get<float>();
        best_valid_ppl = j.at("best_valid_ppl").is_null()
                             ? std::numeric_limits<float>::infinity()
                             : j.at("best_valid_ppl").get<float>();
        best_path = j.at("best_path").get<std::string>();
        if (!std::isfinite(ema_loss) || ema_init != (completed_updates > 0) ||
            (!j.at("best_valid_ppl").is_null() && (!std::isfinite(best_valid_ppl) || best_valid_ppl <= 0)) ||
            (std::isfinite(best_valid_ppl) != !best_path.empty()) ||
            (!best_path.empty() && !std::filesystem::path(best_path).is_absolute())) {
            throw std::runtime_error("Invalid trainer EMA or immutable best reference");
        }
        const auto& saved_timing = j.at("timing");
        if (!saved_timing.is_object() || saved_timing.size() != timing.size()) {
            throw std::runtime_error("Unsupported committed timing schema");
        }
        for (const auto& item : timing.items()) {
            const auto& value = saved_timing.at(item.key());
            if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() < 0) {
                throw std::runtime_error("Invalid committed timing value");
            }
        }
        if (integer(saved_timing.at("updates")) != completed_updates ||
            integer(saved_timing.at("evaluations")) > completed_updates) {
            throw std::runtime_error("Invalid committed timing counters");
        }
        timing = saved_timing;
    }

    void add_time(const char* name, double seconds) {
        timing[name] = timing.at(name).get<double>() + seconds;
    }
};

inline std::string dataset_state(const ops::WikiText2Dataset& train,
                                 const ops::WikiText2Dataset& valid) {
    return Json({{"schema_version", 1}, {"train", train.export_state()},
                 {"valid", valid.export_state()}}).dump();
}

inline auto prepare_restore(TrainerState& trainer, int total_steps, const Json& layout,
                            ops::WikiText2Dataset& train, ops::WikiText2Dataset& valid, int logical_batch) {
    return [&, total_steps, logical_batch](const Json& state, const std::string& text) {
        TrainerState next;
        next.restore(state, total_steps, layout);
        next.physical_batch = logical_batch;
        if (state.contains("physical_batch")) {
            const auto& saved = state.at("physical_batch");
            if (!saved.is_number_integer() || saved < 1 || saved > logical_batch)
                throw std::runtime_error("Invalid physical_batch in checkpoint");
            next.physical_batch = saved.get<int>();
            validate_physical_batch(logical_batch, next.physical_batch);
        }
        const auto j = Json::parse(text);
        if (j.at("schema_version") != 1) throw std::runtime_error("Unsupported CLI dataset envelope");
        auto train_state = train.prepare_import(j.at("train").get<std::string>());
        auto valid_state = valid.prepare_import(j.at("valid").get<std::string>());
        return [&trainer, next = std::move(next), train_state = std::move(train_state),
                valid_state = std::move(valid_state)]() mutable {
            trainer = std::move(next);
            train_state();
            valid_state();
        };
    };
}

template <class Function>
void saving(Function&& function) {
    try {
        function();
    } catch (const std::exception& error) {
        // Fail closed: no ACK or automatic retry after an uncertain durable write.
        throw std::runtime_error(std::string("SAVE_FAILED: ") + error.what() +
                                 "; restart with --resume_state to use the last committed generation");
    }
}

// Legacy adapter writers do not consistently report short writes. Validate their
// complete header, names, shapes and payload length without copying the tensors.
inline void validate_adapter_file(const std::string& path, const ops::NamedParameters& named) {
    std::ifstream in(path, std::ios::binary);
    uint64_t length = 0;
    in.read(reinterpret_cast<char*>(&length), sizeof(length));
    const auto size = std::filesystem::file_size(path);
    if (!in || length > 16 * 1024 * 1024 || size < 8 || length > size - 8) {
        throw std::runtime_error("Incomplete adapter header: " + path);
    }
    std::string header(length, '\0');
    in.read(header.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("Cannot read adapter header: " + path);
    const auto j = Json::parse(header);
    uint64_t bytes = 0;
    for (const auto& parameter : named) {
        const auto& tensor = j.at(parameter.first);
        const auto offsets = tensor.at("data_offsets").get<std::vector<uint64_t>>();
        const uint64_t nbytes = static_cast<uint64_t>(parameter.second->numel()) * sizeof(float);
        if (tensor.at("dtype") != "F32" || tensor.at("shape") != Json(parameter.second->shape()) ||
            offsets.size() != 2 || offsets[1] < offsets[0] || offsets[1] - offsets[0] != nbytes ||
            offsets[1] > size - 8 - length) {
            throw std::runtime_error("Invalid adapter payload: " + parameter.first);
        }
        bytes += nbytes;
    }
    if (size != 8 + length + bytes || j.size() != named.size() + 1) {
        throw std::runtime_error("Incomplete adapter payload: " + path);
    }
}
}  // namespace resume_cli
