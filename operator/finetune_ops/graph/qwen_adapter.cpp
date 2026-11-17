#include "../../third_party/nlohmann/json.hpp"
#include "qwen_model.h"
#include "safetensors_loader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_set>

namespace ops {
namespace {
using Json = nlohmann::json;

std::string adapter_key(size_t layer, const std::string& scope, const std::string& target,
                        const char* ab) {
    return "base_model.model.model.layers." + std::to_string(layer) + "." + scope + "." + target +
           ".lora_" + ab + ".weight";
}

TensorPtr transpose_adapter(const TensorPtr& input) {
    if (!input || input->ndim() != 2 || input->dtype() != kFloat32)
        throw std::invalid_argument("Qwen adapter tensors must be FP32 matrices");
    const int64_t rows = input->shape()[0], cols = input->shape()[1];
    auto result = std::make_shared<Tensor>(std::vector<int64_t>{cols, rows}, kFloat32, kCPU);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) {
            const float value = input->data<float>()[i * cols + j];
            if (!std::isfinite(value))
                throw std::invalid_argument("Non-finite Qwen adapter weight");
            result->data<float>()[j * rows + i] = value;
        }
    return result;
}
} // namespace

// Names follow PEFT; returned tensors retain the native [in,r] / [r,out]
// layout.
std::unordered_map<std::string, TensorPtr> QwenModel::named_lora_parameters() const {
    std::unordered_map<std::string, TensorPtr> result;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& b = const_cast<QwenBlock&>(layers_[i]);
        for (const auto& module : const_cast<QwenModel*>(this)->linear_modules(b))
            if (*module.adapter) {
                const auto& slices = (*module.adapter)->slices();
                if (slices.size() != 1)
                    throw std::logic_error("Expected one Qwen LoRA slice per linear module");
                result[adapter_key(i, module.scope, module.name, "A")] = slices[0].A;
                result[adapter_key(i, module.scope, module.name, "B")] = slices[0].B;
            }
    }
    return result;
}

void QwenModel::save_lora_adapter(const std::string& directory) const {
    namespace fs = std::filesystem;
    if (!lora_rank_)
        throw std::logic_error("Initialize Qwen LoRA before saving an adapter");
    // Validate and transpose before touching output files.
    std::map<std::string, TensorPtr> tensors;
    for (const auto& item : named_lora_parameters())
        tensors[item.first] = transpose_adapter(item.second);
    Json header = Json::object();
    header["__metadata__"] = {{"format", "pt"}};
    uint64_t offset = 0;
    for (const auto& item : tensors) {
        const uint64_t bytes = uint64_t(item.second->numel()) * sizeof(float);
        header[item.first] = {{"dtype", "F32"},
                              {"shape", item.second->shape()},
                              {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    std::string text = header.dump();
    while (text.size() % 8)
        text.push_back(' ');
    fs::create_directories(directory);
    const auto root = fs::path(directory);
    const auto weights_tmp = root / "adapter_model.safetensors.tmp";
    {
        std::ofstream out(weights_tmp, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        const uint64_t size = text.size();
        out.write(reinterpret_cast<const char*>(&size), 8);
        out.write(text.data(), text.size());
        for (const auto& item : tensors)
            out.write(reinterpret_cast<const char*>(item.second->data<float>()),
                      item.second->numel() * sizeof(float));
        out.close();
    }
    Json config = {{"peft_type", "LORA"},
                   {"task_type", "CAUSAL_LM"},
                   {"r", lora_rank_},
                   {"lora_alpha", lora_alpha_},
                   {"lora_dropout", lora_dropout_},
                   {"target_modules", lora_targets_},
                   {"bias", "none"},
                   {"fan_in_fan_out", false},
                   {"inference_mode", true}};
    const auto config_tmp = root / "adapter_config.json.tmp";
    {
        std::ofstream out(config_tmp);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << config.dump(2) << '\n';
        out.close();
    }
    fs::rename(weights_tmp, root / "adapter_model.safetensors");
    fs::rename(config_tmp, root / "adapter_config.json");
}

void QwenModel::load_lora_adapter(const std::string& directory) {
    namespace fs = std::filesystem;
    if (offload_)
        throw std::logic_error("Load adapters before enabling parameter offload");
    std::ifstream input(fs::path(directory) / "adapter_config.json");
    if (!input)
        throw std::runtime_error("Missing adapter_config.json");
    const auto config = Json::parse(input);
    if (config.value("peft_type", std::string()) != "LORA" ||
        config.value("bias", std::string("none")) != "none" ||
        config.value("fan_in_fan_out", false) || config.value("use_dora", false) ||
        config.value("use_rslora", false) || config.value("use_qalora", false) ||
        config.value("lora_bias", false))
        throw std::invalid_argument("Qwen adapter supports standard LoRA with bias=none only");
    for (const auto* key : {"rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters",
                            "layer_replication", "layers_to_transform", "alora_invocation_tokens",
                            "arrow_config", "exclude_modules", "trainable_token_indices"})
        if (config.contains(key) && !config.at(key).is_null() && !config.at(key).empty())
            throw std::invalid_argument(std::string("Unsupported Qwen adapter option: ") + key);
    if (!config.at("r").is_number_integer() || config.at("r") <= 0 || config.at("r") > INT32_MAX)
        throw std::invalid_argument("Qwen adapter rank must be a positive integer");
    const int rank = config.at("r").get<int>();
    const float alpha = config.at("lora_alpha").get<float>();
    const float dropout = config.value("lora_dropout", 0.0f);
    if (rank <= 0 || !std::isfinite(alpha) || alpha <= 0 || !std::isfinite(dropout) ||
        dropout < 0 || dropout >= 1)
        throw std::invalid_argument("Invalid Qwen adapter rank, alpha or dropout");
    if (activation_checkpointing_ && dropout > 0)
        throw std::invalid_argument("Checkpointing requires zero LoRA dropout");
    std::vector<std::string> targets;
    if (config.at("target_modules").is_string()) {
        if (config.at("target_modules") != "all-linear")
            throw std::invalid_argument("Qwen adapter regex target patterns are not supported");
        targets = {"all-linear"};
    } else
        targets = config.at("target_modules").get<std::vector<std::string>>();
    if (targets == std::vector<std::string>{"all-linear"})
        targets = available_targets();
    if (targets.empty())
        throw std::invalid_argument("Qwen adapter has no target modules");
    // PEFT expands all-linear into full module paths when saving. Accept this
    // form only when it selects the same suffixes on every model layer; never
    // silently broaden an adapter that targets just a subset of layers.
    if (targets.front().find("model.layers.") == 0) {
        std::unordered_set<std::string> paths(targets.begin(), targets.end()), suffixes;
        for (const auto& path : targets)
            suffixes.insert(path.substr(path.rfind('.') + 1));
        std::unordered_set<std::string> expected;
        for (size_t i = 0; i < layers_.size(); ++i)
            for (const auto& module : linear_modules(layers_[i]))
                if (suffixes.count(module.name))
                    expected.insert("model.layers." + std::to_string(i) + "." + module.scope + "." +
                                    module.name);
        if (paths != expected || paths.size() != targets.size())
            throw std::invalid_argument("Qwen adapter requires identical targets on every layer");
        targets.assign(suffixes.begin(), suffixes.end());
        std::sort(targets.begin(), targets.end());
    }
    const auto allowed_list = available_targets();
    const std::unordered_set<std::string> allowed(allowed_list.begin(), allowed_list.end());
    std::unordered_set<std::string> unique;
    for (const auto& target : targets)
        if (!allowed.count(target) || !unique.insert(target).second)
            throw std::invalid_argument("Unknown or duplicate Qwen adapter target: " + target);
    SafeTensorsReader reader((fs::path(directory) / "adapter_model.safetensors").string());
    reader.parse_header();
    const auto names = reader.get_tensor_names();
    std::unordered_set<std::string> available(names.begin(), names.end());
    std::unordered_map<std::string, TensorPtr> pending;
    for (size_t i = 0; i < layers_.size(); ++i)
        for (const auto& module : linear_modules(layers_[i])) {
            if (*module.adapter && (*module.adapter)->is_merged())
                throw std::logic_error("Unmerge LoRA before loading an adapter");
            if (!unique.count(module.name))
                continue;
            const auto& shape = (*module.weight)->shape();
            for (const auto* ab : {"A", "B"}) {
                const auto key = adapter_key(i, module.scope, module.name, ab);
                if (!available.count(key))
                    throw std::invalid_argument("Missing Qwen adapter tensor: " + key);
                const std::vector<int64_t> expected = std::string(ab) == "A"
                                                          ? std::vector<int64_t>{rank, shape[0]}
                                                          : std::vector<int64_t>{shape[1], rank};
                if (reader.get_tensor_info(key).shape != expected)
                    throw std::invalid_argument("Qwen adapter tensor shape mismatch: " + key);
                pending[key] = transpose_adapter(reader.load_tensor(key));
            }
        }
    if (pending.size() != names.size())
        throw std::invalid_argument("Unexpected Qwen adapter tensor keys");
    // All external data is checked before changing the current adapter.
    for (auto& b : layers_) {
        for (const auto& module : linear_modules(b))
            module.adapter->reset();
        b.lora_initialized = false;
    }
    lora_rank_ = 0;
    init_lora(rank, alpha, dropout, targets);
    for (const auto& item : named_lora_parameters()) {
        const auto& source = pending.at(item.first);
        std::copy(source->data<float>(), source->data<float>() + source->numel(),
                  item.second->data<float>());
    }
}
} // namespace ops
