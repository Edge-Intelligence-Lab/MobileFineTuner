#pragma once

#include "tokenizer.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace ops {

struct GemmaTokenizerConfig {
    std::string tokenizer_json_path;

    bool add_bos_token = true;
    bool add_eos_token = false;

    std::string bos_token = "<bos>";
    std::string eos_token = "<eos>";
    std::string pad_token = "<pad>";
    std::string unk_token = "<unk>";

    std::vector<std::string> special_tokens = {
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "<start_of_image>",
        "<end_of_image>",
        "<image_soft_token>"
    };

    static GemmaTokenizerConfig from_pretrained(const std::string& model_dir);
};

class GemmaTokenizer {
public:
    explicit GemmaTokenizer(GemmaTokenizerConfig config);
    ~GemmaTokenizer() = default;

    void load();

    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = true,
                            int max_length = 0,
                            bool truncation = true) const;

    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = true) const;

    int get_bos_token_id() const { return bos_token_id_; }
    int get_eos_token_id() const { return eos_token_id_; }
    int get_pad_token_id() const { return pad_token_id_; }
    int get_unk_token_id() const { return unk_token_id_; }
    int get_vocab_size() const { return vocab_size_; }

private:
    GemmaTokenizerConfig config_;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::pair<std::string, std::string>, int, hash_pair> bpe_ranks_;
    std::unordered_set<int> special_token_ids_;

    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int pad_token_id_ = -1;
    int unk_token_id_ = -1;
    int vocab_size_ = 0;

    const std::string sentencepiece_space_ = "\xE2\x96\x81";  // ▁

    void load_tokenizer_json();
    void parse_vocab_block(const std::string& block);
    void parse_merges_block(const std::string& block);

    std::string normalize_text(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& token) const;
    std::pair<int, int> get_best_pair(const std::vector<std::string>& word) const;

    void append_token_id(const std::string& token, std::vector<int>& ids) const;
    void append_byte_fallback(const std::string& token, std::vector<int>& ids) const;
    bool is_byte_fallback_token(const std::string& token) const;
    char decode_byte_token(const std::string& token) const;

    std::string convert_sentencepiece_to_text(const std::string& text) const;
};

}  // namespace ops

