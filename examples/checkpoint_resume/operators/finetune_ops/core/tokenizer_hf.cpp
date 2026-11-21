#include "tokenizer_hf.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef USE_HF_TOKENIZERS
#include <tokenizers_cpp.h>
#endif

namespace ops {

HFTokenizer::HFTokenizer(std::string path, bool is_model_dir)
    : path_(std::move(path)), is_model_dir_(is_model_dir) {}

bool HFTokenizer::load() {
#ifdef USE_HF_TOKENIZERS
    const std::string tok_path = is_model_dir_ ? (path_ + "/tokenizer.json") : path_;
    std::ifstream in(tok_path);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    auto tok = tokenizers::Tokenizer::FromBlobJSON(ss.str());
    if (!tok) {
        return false;
    }
    tokenizer_ = std::move(tok);
    return true;
#else
    return false;
#endif
}

std::vector<int32_t> HFTokenizer::encode(const std::string& text) const {
#ifdef USE_HF_TOKENIZERS
    if (!tokenizer_) {
        throw std::runtime_error("HFTokenizer not loaded");
    }
    return tokenizer_->Encode(text);
#else
    (void)text;
    throw std::runtime_error("HFTokenizer not available (rebuild with -DUSE_HF_TOKENIZERS=ON)");
#endif
}

int32_t HFTokenizer::token_to_id(const std::string& token) const {
#ifdef USE_HF_TOKENIZERS
    if (!tokenizer_) {
        throw std::runtime_error("HFTokenizer not loaded");
    }
    auto id = tokenizer_->TokenToId(token);
    if (id < 0) return -1;
    return static_cast<int32_t>(id);
#else
    (void)token;
    return -1;
#endif
}

}  // namespace ops
