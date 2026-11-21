/**
 * @file wikitext2_dataset.h
 * @brief WikiText-2 dataset loader (concatenate + chunk, aligned with HF)
 */

#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer_hf.h"
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <array>

namespace ops {

struct WT2Config {
    // Absolute paths
    std::string train_path;
    std::string valid_path;
    std::string test_path;
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string jsonl_test;
    int jsonl_vocab_limit = 0; // Session trainers require validated token/mask rows.
    std::string pretokenized_path;    // Offline tokenized token stream (.bin)
    std::string pretokenized_meta;    // Metadata json (optional)

    int seq_len = 256;         // Training sequence length (<=1024)
    int stride  = -1;          // -1 means equal to seq_len (no overlap); otherwise use sliding window stride
    int eos_id  = 50256;       // GPT-2 <|endoftext|>
    int pad_id  = 0;           // PAD fill value
    bool insert_eos_between_lines = true;   // Insert EOS between samples
    bool drop_last = true;     // true for training; validation can set false to keep tail
    uint64_t seed = 2025;      // Shuffle/sampling random seed
    bool shuffle_train = true; // Whether to shuffle training order
    
    // Streaming loading configuration (memory optimization)
    bool streaming_mode = true;       // Whether to use streaming loading (no full token residency)
    size_t max_cache_tokens = 100000; // Maximum cached tokens (streaming mode)
    float data_fraction = 1.0f;       // Data fraction to use (0~1]
};

enum class Split { Train, Valid, Test };

struct Batch {
    TensorPtr input_ids;       // [B, S] int32
    TensorPtr attention_mask;  // [B, S] float32 (1=valid, 0=pad)
    TensorPtr labels;          // [B, S] int32 (pad->-100)
};

class WikiText2Dataset {
public:
    WikiText2Dataset(const WT2Config& cfg, HFTokenizer* tok);
    WikiText2Dataset(const WT2Config& cfg,
                     std::function<std::vector<int32_t>(const std::string&)> encode_fn);
    
    /**
     * @brief Read file -> tokenize -> concatenate -> build index
     */
    void load(Split split);
    
    /**
     * @brief Number of available chunks
     */
    size_t num_sequences() const;
    
    /**
     * @brief Shuffle order (only used for Train)
     */
    void shuffle();
    
    /**
     * @brief Get a batch starting from index_start
     */
    Batch get_batch(size_t index_start, size_t batch_size) const;
    
    /**
     * @brief Convenience interface: auto-increment cursor (loops back or truncates at end)
     */
    Batch next_batch(size_t batch_size, bool need_loop = true);
    
    /**
     * @brief Get first few tokens (for sanity check)
     */
    std::vector<int32_t> peek_tokens(size_t count) const;
    
    /**
     * @brief Reset cursor
     */
    void reset_cursor();

    // Load the same immutable data/tokenizer first. Restore never reshuffles.
    // Paths and token contents are verified by the caller's asset manifest.
    std::string export_state() const;
    void import_state(const std::string& state);
    std::function<void()> prepare_import(const std::string& state);

private:
    struct PretokenizedSplit {
        size_t offset = 0;
        size_t length = 0;
        bool available = false;
    };

    struct PretokenizedMeta {
        bool loaded = false;
        std::string meta_path;
        size_t total_tokens = 0;
        int32_t eos_id = -1;
        int32_t pad_id = 0;
        int32_t bos_id = -1;
        int32_t unk_id = -1;
        int32_t vocab_size = 0;
        bool insert_eos_between_lines = true;
        PretokenizedSplit train;
        PretokenizedSplit valid;
        PretokenizedSplit test;
    };

    // Read text lines for specified split
    std::vector<std::string> read_lines_for_split(Split split) const;
    
    // Convert lines to token stream (concatenate + insert EOS)
    std::vector<int32_t> tokenize_and_pack(const std::vector<std::string>& lines) const;
    
    // Generate chunk start indices based on stride/seq_len
    void build_chunk_indices(const std::vector<int32_t>& ids);
    
    // True streaming loading: read window from file on demand
    void load_window_from_file(size_t global_token_start, size_t num_tokens);
    
    // Precompute true token offsets for all chunks
    void precompute_chunk_offsets();
    
    // Pretokenized mode: read metadata/split
    void ensure_pretokenized_meta_loaded();
    void load_pretokenized_split(Split split);

    WT2Config cfg_;
    bool loaded_ = false;
    Split loaded_split_ = Split::Train;
    std::function<std::vector<int32_t>(const std::string&)> encode_fn_;

    // Streaming mode: file path instead of full text
    std::string current_file_path_;   // Current split file path
    std::vector<int32_t> ids_;        // Current window token cache (only ~100k)
    size_t ids_global_offset_ = 0;    // Which global token ids_[0] corresponds to
    size_t total_tokens_ = 0;         // Total global token count (from prescan)
    
    // True starting point of each chunk (global token offset)
    std::vector<size_t> starts_;      // Length M = num_sequences()

    // Indices sampled for batch (Train can shuffle)
    std::vector<size_t> order_;       // Length M
    mutable size_t cursor_;           // Starting position of next batch in order_
    mutable std::mt19937_64 rng_;
    
    // Batch buffer reuse (avoid creating new Tensor each time)
    mutable std::vector<int32_t> batch_input_buffer_;
    mutable std::vector<int32_t> batch_label_buffer_;
    mutable std::vector<float> batch_attn_buffer_;
    
    // Pretokenized mode
    mutable PretokenizedMeta pretokenized_meta_;
    bool pretokenized_mode_ = false;
    bool jsonl_mode_ = false;
    std::vector<std::vector<int32_t>> jsonl_ids_;
    std::vector<std::vector<int32_t>> jsonl_mask_;

    void load_jsonl_split(Split split);
};

}  // namespace ops
