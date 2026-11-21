#pragma once

#include <string>

#include "qwen_model.h"

namespace ops {

struct QwenLoraMetadata {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;
    bool qv_only = true;
};

class QwenLoraSaver {
public:
    static void save_safetensors(const std::string& path,
                                 const QwenModel& model,
                                 const QwenLoraMetadata& meta);

    static bool parse_metadata(const std::string& path, QwenLoraMetadata& meta);

    static void load_safetensors(const std::string& path,
                                 QwenModel& model,
                                 const QwenLoraMetadata& meta);
};

}  // namespace ops
