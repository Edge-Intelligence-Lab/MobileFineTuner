// HuggingFace tokenizers (C++ binding) wrapper
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_HF_TOKENIZERS
#include <tokenizers_cpp.h>
#endif

namespace ops {

class HFTokenizer {
public:
    explicit HFTokenizer(std::string path, bool is_model_dir = true);
    ~HFTokenizer() = default;

    // Load tokenizer.json (returns true on success)
    bool load();

    // Encode text into token IDs (no special tokens appended)
    std::vector<int32_t> encode(const std::string& text) const;

    // Resolve token id by string (returns -1 if missing)
    int32_t token_to_id(const std::string& token) const;

private:
    std::string path_;
    bool is_model_dir_ = true;

#ifdef USE_HF_TOKENIZERS
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
#endif
};

}  // namespace ops
