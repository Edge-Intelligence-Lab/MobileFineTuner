#pragma once

#include "finetune_ops/graph/model_registry.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include <cctype>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void configure_qwen_base_weight_storage(ops::SafeTensorsLoadOptions& options,
                                               const std::string& policy) {
    if (policy != "fp32" && policy != "auto" && policy != "native")
        throw std::invalid_argument("--base_weight_storage must be auto, native or fp32");
    // QwenModel promotes norms, biases and convolution to FP32 after loading;
    // all frozen matrices, including hybrid mixer projections, can stay native.
    options.auto_promote_fp16 = policy == "fp32";
}

inline std::vector<std::string> parse_qwen_lora_targets(const std::string& text) {
    if (text.empty() || text.back() == ',')
        throw std::invalid_argument("LoRA targets must be nonempty comma-separated module names");
    std::vector<std::string> targets;
    std::stringstream stream(text);
    std::string target;
    while (std::getline(stream, target, ',')) {
        const auto begin = target.find_first_not_of(" \t\r\n");
        const auto end = target.find_last_not_of(" \t\r\n");
        if (begin == std::string::npos)
            throw std::invalid_argument("Empty LoRA target");
        targets.push_back(target.substr(begin, end - begin + 1));
    }
    return targets;
}

// Both dataset frontends share the same storage and execution contract.
struct QwenExecutionOptions {
    std::string base_weight_storage = "auto";
    bool activation_checkpointing = false;
    bool memory_efficient_attention = false;
    std::string offload_dir;
    int offload_resident_mb = 1536;

    bool parse(const std::string& key, int& index, int argc, char** argv) {
        auto value = [&]() -> std::string {
            if (index + 1 >= argc)
                throw std::invalid_argument("arg missing val: " + key);
            return argv[++index];
        };
        if (key == "--base_weight_storage")
            base_weight_storage = value();
        else if (key == "--activation_checkpointing")
            activation_checkpointing = true;
        else if (key == "--memory_efficient_attention")
            memory_efficient_attention = true;
        else if (key == "--offload_dir")
            offload_dir = value();
        else if (key == "--offload_resident_mb")
            offload_resident_mb = std::stoi(value());
        else
            return false;
        return true;
    }
    void validate(float dropout, bool synthetic = false) const {
        ops::SafeTensorsLoadOptions storage;
        configure_qwen_base_weight_storage(storage, base_weight_storage);
        const bool memory =
            activation_checkpointing || memory_efficient_attention || !offload_dir.empty();
        if (synthetic && memory)
            throw std::invalid_argument(
                "memory options require real training; synthetic_smoke does not apply them");
        if ((activation_checkpointing || !offload_dir.empty()) && dropout != 0)
            throw std::invalid_argument("memory training examples require --lora_dropout 0");
        if (offload_resident_mb <= 0)
            throw std::invalid_argument("--offload_resident_mb must be positive");
        if (!offload_dir.empty() && base_weight_storage != "fp32")
            throw std::invalid_argument("offload requires --base_weight_storage fp32");
    }
    template <class Output> void report_offload(const ops::QwenModel& model, Output& output) const {
        if (offload_dir.empty())
            return;
        const auto stats = model.sharding_stats();
        output << "[Offload] entries=" << stats.entries << " loads=" << stats.loads
               << " evictions=" << stats.evictions << " bytes_read=" << stats.bytes_read
               << " bytes_written=" << stats.bytes_written << " writes=" << stats.writes
               << std::endl;
    }
    void apply(ops::QwenModel& model) const {
        // Resume can restore a different dropout than the CLI's initial value.
        validate(model.lora_dropout());
        const auto capability = ops::ModelRegistry::capabilities_for(model.config().model_type);
        if ((memory_efficient_attention && !capability.streamed_attention) ||
            (activation_checkpointing && !capability.activation_checkpointing) ||
            (!offload_dir.empty() && !capability.frozen_parameter_offload))
            throw std::invalid_argument(
                "Requested Qwen execution option is unavailable in this library build");
        model.set_memory_efficient_attention(memory_efficient_attention);
        model.set_activation_checkpointing(activation_checkpointing);
        if (!offload_dir.empty()) {
            ops::sharding::ShardConfig config;
            config.offload_dir = offload_dir;
            config.max_resident_bytes = static_cast<size_t>(offload_resident_mb) * 1024 * 1024;
            model.enable_parameter_sharding(config);
        }
    }
};
