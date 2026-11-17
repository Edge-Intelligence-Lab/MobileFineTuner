#pragma once

#include "finetune_ops/graph/safetensors_loader.h"
#include <cctype>
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
