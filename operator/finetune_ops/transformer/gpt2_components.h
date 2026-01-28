/**
 * @file gpt2_components.h
 * @brief GPT-2 model components and configuration
 * 
 * This file defines the core components of the GPT-2 model including
 * configuration structures, transforer blocks, and model architecture.
 * It providess both the base GPT-2 implementsation and fine-tuning variants.
 */

#pragma once

#include "tensor.h"
#include "basic_ops.h"
#include "attention.h"
#include <memory>
#include <vector>

namespace ops {

/**
 * @brief Configuration structure for GPT-2 model
 * 
 * Contains all hyperparameters and architectural choices for GPT-2.
 * Supports different model sizes from 117M to 1.5B parameters.
 */
struct GPT2Config {
    int vocab_size = 50257;           /**< Vocabulary size */
    int n_positions = 1024;           /**< Maximum sequence length */
    int n_embd = 768;                 /**< Embedding dimension */
    int n_layer = 12;                 /**< Number of transforer layers */
    int n_head = 12;                  /**< Number of attention heads */
    int n_inner = 3072;               /**< MLP intermediate dimension */
    float layer_norm_epsilon = 1e-5f; /**< Layer normalization epsilon */

    // Inference optimization settings
    bool use_kv_cache = true;         /**< Enable KV caching for inference */
    int max_seq_length = 2048;        /**< Maximum sequence length for inference */
    float dropout = 0.1f;             /**< Dropout rate */

    // Constructors
    GPT2Config() = default;
    
    /**
     * @brief Construct GPT-2 configuration with custom parameters
     * @param vocab Vocabulary size
     * @param pos Maximum sequence length
     * @param embd Embedding dimension
     * @param layer Number of transforer layers
     * @param head Number of attention heads
     * @param inner MLP intermediate dimension
     */
    GPT2Config(int vocab, int pos, int embd, int layer, int head, int inner)
        : vocab_size(vocab), n_positions(pos), n_embd(embd),
          n_layer(layer), n_head(head), n_inner(inner) {}

    /**
     * @brief Get GPT-2 117M model configuration
     * @return Configuration for 117M parameter model
     */
    static GPT2Config GPT2_117M() {
        return GPT2Config(50257, 1024, 768, 12, 12, 3072);
    }

    static GPT2Config GPT2_345M() {
        return GPT2Config(50257, 1024, 1024, 24, 16, 4096);
    }

    static GPT2Config GPT2_774M() {
        return GPT2Config(50257, 1024, 1280, 36, 20, 5120);
    }

    static GPT2Config GPT2_1558M() {
        return GPT2Config(50257, 1024, 1600, 48, 25, 6400);
    }
};

class LayerNorm {
private:
    std::unique_ptr<Tensor> weight_;
    std::unique_ptr<Tensor> bias_;
    float epsilon_;

public:
    LayerNorm(int n_embd, float epsilon = 1e-5f);
    ~LayerNorm() = default;

    std::unique_ptr<Tensor> forward(const Tensor& input);
    void print_info() const;
};

class FeedForward {
private:
    std::unique_ptr<Tensor> fc1_weight_;
    std::unique_ptr<Tensor> fc1_bias_;
    std::unique_ptr<Tensor> fc2_weight_;
    std::unique_ptr<Tensor> fc2_bias_;

public:
    FeedForward(int n_embd, int n_inner);
    ~FeedForward() = default;

    std::unique_ptr<Tensor> forward(const Tensor& input);
    void print_info() const;
};

class Embedding {
private:
    std::unique_ptr<Tensor> weight_;
    int vocab_size_;
    int n_embd_;

public:
    Embedding(int vocab_size, int n_embd);
    ~Embedding() = default;

    std::unique_ptr<Tensor> forward(const std::vector<int>& input_ids);
    void print_info() const;
};

class TransforerBlock {
private:
    std::unique_ptr<LayerNorm> ln1_;
    std::unique_ptr<LayerNorm> ln2_;
    std::unique_ptr<FeedForward> mlp_;

    std::unique_ptr<Tensor> attn_qkv_weight_;
    std::unique_ptr<Tensor> attn_qkv_bias_;
    std::unique_ptr<Tensor> attn_proj_weight_;
    std::unique_ptr<Tensor> attn_proj_bias_;

    int n_head_;
    int n_embd_;

public:
    TransforerBlock(const GPT2Config& config);
    ~TransforerBlock() = default;

    std::unique_ptr<Tensor> forward(const Tensor& hidden_states);
    void print_info() const;
};

}