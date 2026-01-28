/**
 * @file wikitext2_dataset.h
 * @brief WikiText-2 dataset loader (concatenate + chunk, HF aligned)
 */

#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer_bpe.h"
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
    // Optional: JSONL pretokenized causal LM data with masks: {"ids":[...], "mask":[...]}
    // If any jsonl_* provided, JSONL mode is used (raw/pretokenized ignored)
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string jsonl_test;
    std::string pretokenized_path;    // offline token stream (.bin)
    std::string pretokenized_meta;    // meta json (optional)

    int seq_len = 256;         // training sequence length (<=1024)
    int stride  = -1;          // -1 means seq_len (no overlap); otherwise sliding stride
    int eos_id  = 50256;       // GPT-2 <|endoftext|>
    int pad_id  = 0;           // pad value
    bool insert_eos_between_lines = true;   // insert EOS between samples
    bool drop_last = true;     // train=true; valid can set false to keep tail
    uint64_t seed = 2025;      // RNG seed for shuffling/sampling
    bool shuffle_train = true; // shuffle training order
    
    // Streaming load (memory friendly)
    bool streaming_mode = true;       // stream tokens instead of loading all
    size_t max_cache_tokens = 100000; // max cached tokens in streaming mode
    float data_fraction = 1.0f;       // fraction of data to use [0,1]
};

enum class Split { Train, Valid, Test };

struct Batch {
    TensorPtr input_ids;       // [B, S] int32
    TensorPtr attention_mask;  // [B, S] float32 (1=valid, 0=pad)
    TensorPtr labels;          // [B, S] int32 (pad→-100)
};

class WikiText2Dataset {
public:
    WikiText2Dataset(const WT2Config& cfg, GPT2BPETokenizer* tok);
    WikiText2Dataset(const WT2Config& cfg,
                     std::function<std::vector<int32_t>(const std::string&)> encode_fn);
    
    /**
     * @brief Load file → tokenize → concatenate → index
     */
    void load(Split split);
    
    /**
     * @brief Number of available chunks
     */
    size_t num_sequences() const;
    
    /**
     * @brief Shuffle order (train only)
     */
    void shuffle();
    
    /**
     * @brief Get a batch starting at index_start
     */
    Batch get_batch(size_t index_start, size_t batch_size) const;
    
    /**
     * @brief Convenience: auto-increment cursor (loop or stop at end)
     */
    Batch next_batch(size_t batch_size, bool need_loop = true);
    
    /**
     * @brief Peek first tokens (sanity check)
     */
    std::vector<int32_t> peek_tokens(size_t count) const;
    
    /**
     * @brief Reset cursor
     */
    void reset_cursor();

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

    // Read lines for a split
    std::vector<std::string> read_lines_for_split(Split split) const;
    
    // Lines → token stream (concatenate + insert EOS)
    std::vector<int32_t> tokenize_and_pack(const std::vector<std::string>& lines) const;
    
    // Build chunk start indices based on stride/seq_len
    void build_chunk_indices(const std::vector<int32_t>& ids);
    
    // Streaming: load window on demand from file
    void load_window_from_file(size_t global_token_start, size_t num_tokens);
    
    // Precompute true token offsets for all chunks
    void precompute_chunk_offsets();
    
    // Pretokenized mode: load metadata / slice
    void ensure_pretokenized_meta_loaded();
    void load_pretokenized_split(Split split);
    
    // JSONL mode: each line is {"ids":[...], "mask":[...]} (length <= seq_len)
    void load_jsonl_split(Split split);
    static bool parse_jsonl_ids_mask(const std::string& line,
                                     std::vector<int32_t>& ids_out,
                                     std::vector<uint8_t>& mask_out);

    WT2Config cfg_;
    GPT2BPETokenizer* tok_;   // non-owning pointer
    std::function<std::vector<int32_t>(const std::string&)> encode_fn_;

    // Streaming mode state
    std::string current_file_path_;   // current split file path
    std::vector<int32_t> ids_;        // token cache for current window (~1e5)
    size_t ids_global_offset_ = 0;    // global token index for ids_[0]
    size_t total_tokens_ = 0;         // total tokens (pre-scan)
    
    // Chunk start offsets (global token indices)
    std::vector<size_t> starts_;      // length M = num_sequences()
    
    // Order for batching (train may shuffle)
    std::vector<size_t> order_;       // length M
    mutable size_t cursor_;           // next batch start position in order_
    mutable std::mt19937_64 rng_;
    
    // Batch buffer reuse (avoid realloc each time)
    mutable std::vector<int32_t> batch_input_buffer_;
    mutable std::vector<int32_t> batch_label_buffer_;
    mutable std::vector<float> batch_attn_buffer_;
    
    // Pretokenized mode
    mutable PretokenizedMeta pretokenized_meta_;
    bool pretokenized_mode_ = false;
    
    // JSONL mask mode
    bool jsonl_mode_ = false;
    struct JsonlSample {
        std::vector<int32_t> ids;
        std::vector<uint8_t> mask;
    };
    std::vector<JsonlSample> jsonl_samples_;
};

}  // namespace ops
