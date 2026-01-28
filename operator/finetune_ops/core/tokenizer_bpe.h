/**
 * @file tokenizer_bpe.h
 * @brief GPT-2 Byte-Level BPE Tokenizer (aligned with HuggingFace)
 *
 * Implements standard GPT-2 BPE with support for:
 * - Loading vocab.json / merges.txt / special_tokens_map.json
 * - Byte-level mapping (256 bytes → unicode codepoints)
 * - BPE merges in merges.txt rank order
 * - Encode/decode behavior aligned with HuggingFace transformers
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ops {

/**
 * @brief GPT-2 Byte-Level BPE configuration
 */
struct BPEConfig {
    std::string vocab_path;           // path to vocab.json
    std::string merges_path;          // path to merges.txt
    std::string special_tokens_path;  // optional path to special_tokens_map.json
    
    int eos_token_id = 50256;         // <|endoftext|>
    int bos_token_id = 50256;         // GPT-2 typically has no separate BOS
    int pad_token_id = 50256;         // reuse eos (HF default)
    int unk_token_id = 50256;         // rarely used for GPT-2
    
    bool add_prefix_space = false;    // GPT-2 default: no leading space
    
    BPEConfig() = default;
    
    // Load from a model directory with standard filenames
    static BPEConfig from_pretrained(const std::string& model_dir) {
        BPEConfig cfg;
        cfg.vocab_path = model_dir + "/vocab.json";
        cfg.merges_path = model_dir + "/merges.txt";
        cfg.special_tokens_path = model_dir + "/special_tokens_map.json";
        return cfg;
    }
};

/**
 * @brief GPT-2 Byte-Level BPE Tokenizer
 *
 * Aligned with HuggingFace GPT2Tokenizer:
 * - Byte→Unicode mapping (256 bytes → 256 chars, skipping unprintable)
 * - BPE merges follow merges.txt rank order
 * - Special token: <|endoftext|> (id=50256)
 */
class GPT2BPETokenizer {
public:
    explicit GPT2BPETokenizer(const BPEConfig& config);
    ~GPT2BPETokenizer() = default;
    
    /**
     * @brief Load tokenizer assets (vocab/merges/special_tokens)
     */
    void load();
    
    /**
     * @brief Encode text into token IDs
     * @param text raw text
     * @param add_special_tokens auto-append <|endoftext|> (default false)
     * @param max_length max length (0 = unlimited)
     * @param truncation whether to truncate when exceeding max_length
     * @return token IDs
     */
    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = false,
                            int max_length = 0,
                            bool truncation = true);
    
    /**
     * @brief Decode token IDs back to text
     * @param ids token IDs
     * @param skip_special_tokens skip special tokens (default false)
     * @return original text
     */
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = false);
    
    /**
     * @brief Batch encode (with padding/truncation)
     * @param texts list of texts
     * @param max_length max length (0 = unlimited)
     * @param padding padding mode ("max_length" | "longest" | "none")
     * @param truncation whether to truncate
     * @return {input_ids, attention_mask}
     */
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
    batch_encode(const std::vector<std::string>& texts,
                 int max_length = 0,
                 const std::string& padding = "longest",
                 bool truncation = true);
    
    // Access special token IDs
    int get_eos_token_id() const { return config_.eos_token_id; }
    int get_bos_token_id() const { return config_.bos_token_id; }
    int get_pad_token_id() const { return config_.pad_token_id; }
    int get_unk_token_id() const { return config_.unk_token_id; }
    int get_vocab_size() const { return vocab_size_; }
    
    // Utility
    std::string get_token_string(int token_id) const;

private:
    BPEConfig config_;
    
    // Byte→Unicode mapping (256 entries)
    std::unordered_map<uint8_t, std::string> byte_encoder_;    // byte → unicode UTF-8 char
    std::unordered_map<std::string, uint8_t> byte_decoder_;    // unicode UTF-8 char → byte
    
    // Vocab and reverse mapping
    std::unordered_map<std::string, int> vocab_;  // token → id
    std::unordered_map<int, std::string> id_to_token_;  // id → token
    int vocab_size_;
    
    // BPE merges (pair → rank)
    std::unordered_map<std::string, int> bpe_ranks_;  // "a b" → rank
    
    // Special token cache
    std::unordered_map<std::string, int> special_tokens_;
    
    // Internal helpers
    void build_byte_encoder();                       // build byte↔unicode mapping
    void load_vocab();                               // load vocab.json
    void load_merges();                              // load merges.txt
    void load_special_tokens();                      // load special_tokens_map.json
    
    std::string bytes_to_unicode(const std::string& text);          // text → byte-level unicode
    std::string unicode_to_bytes(const std::string& unicode_text);  // reverse mapping
    
    std::vector<std::string> bpe(const std::string& token);         // BPE merge on a single token
    std::pair<int, int> get_best_pair(const std::vector<std::string>& word);  // best-ranked pair
    std::vector<std::string> split_to_words(const std::string& text);         // whitespace-based split
};

/**
 * @brief Qwen Byte-Level BPE configuration
 *
 * Assets:
 * - vocab.json / merges.txt
 * - tokenizer.json (added_tokens and specials)
 * - config.json (final vocab_size for embedding alignment)
 */
struct QwenTokenizerConfig {
    std::string vocab_path;
    std::string merges_path;
    std::string tokenizer_json_path;   // parse added_tokens (incl. specials)
    std::string config_json_path;      // read vocab_size for embedding alignment

    int eos_token_id = 151643;         // <|endoftext|>
    int bos_token_id = -1;             // Qwen does not explicitly use BOS
    int pad_token_id = 151643;         // same as HF: pad = eos
    int unk_token_id = -1;             // HF does not define unk

    // If config vocab_size exceeds vocab count, override to match embedding size
    int vocab_size_override = -1;

    static QwenTokenizerConfig from_pretrained(const std::string& model_dir) {
        QwenTokenizerConfig cfg;
        cfg.vocab_path = model_dir + "/vocab.json";
        cfg.merges_path = model_dir + "/merges.txt";
        cfg.tokenizer_json_path = model_dir + "/tokenizer.json";
        cfg.config_json_path = model_dir + "/config.json";
        return cfg;
    }
};

/**
 * @brief Qwen Byte-Level BPE Tokenizer (aligned with HuggingFace Qwen2Tokenizer)
 *
 * Supports:
 * - Loading vocab/merges (byte-level BPE)
 * - Parsing tokenizer.json added_tokens (incl. specials)
 * - Handling <|im_start|> etc. to avoid unintended BPE splits
 * - vocab_size override to match embedding size (e.g., 151936)
 */
class QwenBPETokenizer {
public:
    explicit QwenBPETokenizer(const QwenTokenizerConfig& cfg);
    ~QwenBPETokenizer() = default;

    void load();

    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = false,
                            int max_length = 0,
                            bool truncation = true);

    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = false);

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
    batch_encode(const std::vector<std::string>& texts,
                 int max_length = 0,
                 const std::string& padding = "longest",
                 bool truncation = true);

    int get_eos_token_id() const { return eos_token_id_; }
    int get_bos_token_id() const { return bos_token_id_; }
    int get_pad_token_id() const { return pad_token_id_; }
    int get_vocab_size() const { return vocab_size_; }

    std::string get_token_string(int token_id) const;

private:
    QwenTokenizerConfig config_;

    // Byte→Unicode mapping
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    // Vocab & merges
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> bpe_ranks_;

    // added_tokens (includes non-special tokens)
    std::unordered_map<std::string, int> added_tokens_;
    std::unordered_map<int, std::string> id_to_added_;
    std::vector<std::string> added_tokens_sorted_;  // sorted by length desc for prefix matching
    std::unordered_map<std::string, bool> added_special_flags_;

    // vocab_size (may exceed actual vocab, follows config.json)
    int vocab_size_ = 0;
    int eos_token_id_ = 151643;
    int bos_token_id_ = -1;
    int pad_token_id_ = 151643;

    // Internal helpers
    void build_byte_encoder();
    void load_vocab();
    void load_merges();
    void load_added_tokens();
    void load_vocab_size_override();

    std::string bytes_to_unicode(const std::string& text);
    std::string unicode_to_bytes(const std::string& unicode_text);

    std::vector<std::string> bpe(const std::string& token);
    std::pair<int, int> get_best_pair(const std::vector<std::string>& word);

    // Pre-tokenization: mimic tokenizer.json regex (no external deps; simple Unicode classes)
    std::vector<std::string> split_to_words(const std::string& text);

    // Slice text by added_tokens (prevent BPE on added tokens)
    struct Segment {
        std::string text;
        bool is_added;
    };
    std::vector<Segment> split_by_added_tokens(const std::string& text);
};

}  // namespace ops
