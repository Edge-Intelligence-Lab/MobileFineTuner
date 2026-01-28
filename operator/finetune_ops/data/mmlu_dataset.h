/**
 * @file mmlu_dataset.h
 * @brief MMLU dataset loader - MCQ to text + masked causal LM
 * 
 * Format: convert CSV multiple-choice into text:
 * Question: {question}
 * A. {choice_a}
 * B. {choice_b}
 * C. {choice_c}
 * D. {choice_d}
 * Answer: {answer}
 * 
 * Masked loss: compute loss only on the "Answer: X" token(s)
 */

#pragma once

#include "../core/tokenizer_bpe.h"
#include "../core/tensor.h"
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <unordered_set>

namespace ops {

struct MMLUConfig {
    std::string data_dir;           // MMLU data root (contains dev/test/val)
    int seq_len = 128;              // sequence length
    int batch_size = 8;             // batch size
    uint64_t seed = 2025;           // RNG seed
    
    // Optional: restrict to specific subjects (empty = all)
    std::vector<std::string> subjects;  // empty = all subjects
    
    // Few-shot configuration
    int num_shots = 0;              // 0=zero-shot, 5=5-shot
    bool use_dev_as_train = true;   // use dev split for training (e.g., 5 per subject)
};

// Single MMLU sample
struct MMLUSample {
    std::string question;
    std::string choices[4];         // A, B, C, D
    char answer;                    // 'A', 'B', 'C', 'D'
    std::string subject;            // subject name
    
    // Convert to training text
    std::string to_prompt() const;
    std::string to_full_text() const;  // prompt + answer
};

// Batch structure (aligned with WikiText2 dataset)
struct MMLUBatch {
    TensorPtr input_ids;       // [B, S] int32
    TensorPtr attention_mask;  // [B, S] float32 (1=valid, 0=pad)
    TensorPtr labels;          // [B, S] int32 (-100 = ignore)
    
    size_t num_samples = 0;    // actual sample count
};

class MMLUDataset {
public:
    MMLUDataset(const MMLUConfig& cfg, QwenBPETokenizer* tok);
    
    /**
     * @brief Load training data (dev split)
     */
    void load_train();
    
    /**
     * @brief Load test data (test split)
     */
    void load_test();
    
    /**
     * @brief Load validation data (val split)
     */
    void load_val();
    
    /**
     * @brief Total number of samples
     */
    size_t num_samples() const { return samples_.size(); }
    
    /**
     * @brief Total number of batches
     */
    size_t num_batches() const;
    
    /**
     * @brief Shuffle sample order
     */
    void shuffle();
    
    /**
     * @brief Get a batch
     * @param batch_idx batch index
     * @return MMLUBatch (masked labels)
     */
    MMLUBatch get_batch(size_t batch_idx) const;
    
    /**
     * @brief Get the next batch sequentially
     * @param loop whether to loop
     */
    MMLUBatch next_batch(bool loop = true);
    
    /**
     * @brief Reset cursor
     */
    void reset_cursor() { cursor_ = 0; }
    
    /**
     * @brief Get list of subjects
     */
    std::vector<std::string> get_subjects() const;
    
    /**
     * @brief Print dataset statistics
     */
    void print_stats() const;

private:
    // Load samples from CSV
    void load_from_dir(const std::string& subdir);
    
    // Parse a CSV line
    MMLUSample parse_csv_line(const std::string& line, const std::string& subject);
    
    // Encode one sample to input_ids and labels
    // Only answer tokens keep labels; others set to -100
    void encode_sample(const MMLUSample& sample,
                      std::vector<int32_t>& input_ids,
                      std::vector<int32_t>& labels) const;
    
    // Locate answer token position
    int find_answer_token_position(const std::vector<int32_t>& full_ids,
                                   const std::vector<int32_t>& prompt_ids) const;

    MMLUConfig cfg_;
    QwenBPETokenizer* tok_;
    
    std::vector<MMLUSample> samples_;
    std::vector<size_t> order_;         // shuffled indices
    mutable size_t cursor_ = 0;
    mutable std::mt19937_64 rng_;
    
    // Pre-encoded tokens (avoid repeated encoding)
    mutable std::vector<int32_t> cached_prompt_prefix_;  // "Question: "
    mutable std::vector<int32_t> cached_answer_prefix_;  // "\nAnswer: "
    mutable bool cache_initialized_ = false;
    
    void init_token_cache() const;
};

}  // namespace ops

