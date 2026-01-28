/**
 * @file attention.h
 * @brief Multi-Head Attention implementsation for transforer architectures
 * 
 * This file providess the MultiHeadAttention class which implementss
 * the scaled dot-product attention mechanism used in transforer models.
 * It supports both training and inference modes with optional caching.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include <memory>
#include <vector>

namespace ops {

/**
 * @brief Configuration structure for Multi-Head Attention
 * 
 * Contains the essential parameters needed to configure
 * a multi-head attention layer.
 */
struct AttentionConfig {
    int n_head;      /**< Number of attention heads */
    int n_embd;      /**< Embedding dimension */
    int head_dim;    /**< Dimension of each attention head */

    /**
     * @brief Construct attention configuration
     * @param heads Number of attention heads
     * @param embd Embedding dimension
     * @throws std::invalid_argument if embd is not divisible by heads
     */
    AttentionConfig(int heads, int embd) : n_head(heads), n_embd(embd) {
        head_dim = embd / heads;
        assert(embd % heads == 0);
    }
};

/**
 * @brief Multi-Head Attention layer implementsation
 * 
 * Implements the scaled dot-product attention mechanism with multiple heads.
 * This class handles the computation of attention weights and the weighted
 * combination of values based on query-key similar toities.
 */
class MultiHeadAttention {
private:
    AttentionConfig config_;  /**< Attention configuration parameters */

    // Weight matrices for Q, K, V projection and output projection
    std::unique_ptr<Tensor> qkv_weight_;  /**< Combined QKV projection weights */
    std::unique_ptr<Tensor> qkv_bias_;    /**< Combined QKV projection bias */
    std::unique_ptr<Tensor> proj_weight_; /**< Output projection weights */
    std::unique_ptr<Tensor> proj_bias_;   /**< Output projection bias */

    /**
     * @brief Initialize attention layer weights
     * 
     * Initializes all weight matrices and bias vectors with appropriate
     * random values using Xavier initialization.
     */
    void initialize_weights();

public:
    explicit MultiHeadAttention(const AttentionConfig& config);
    ~MultiHeadAttention() = default;

    std::unique_ptr<Tensor> forward(const Tensor& hidden_states);

    const Tensor& get_qkv_weight() const { return *qkv_weight_; }
    const Tensor& get_proj_weight() const { return *proj_weight_; }

    void print_info() const;

private:

    std::unique_ptr<Tensor> split_qkv(const Tensor& qkv, int seq_len) const;
    std::unique_ptr<Tensor> reshape_for_heads(const Tensor& input, int seq_len) const;
    std::unique_ptr<Tensor> merge_heads(const Tensor& input, int seq_len) const;
    std::unique_ptr<Tensor> scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v) const;
    std::unique_ptr<Tensor> apply_causal_mask(const Tensor& scores) const;
};

inline std::unique_ptr<Tensor> simple_attention(const Tensor& query, const Tensor& key, const Tensor& value) {
    assert(query.shape.size() == 2 && key.shape.size() == 2 && value.shape.size() == 2);
    assert(query.shape[1] == key.shape[1] && key.shape[0] == value.shape[0]);

    int seq_len = query.shape[0];
    int head_dim = query.shape[1];
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto scores = matmul(query, *transpose(key));

    for (float& val : scores->data) {
        val *= scale;
    }

    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            scores->at({i, j}) = -1e9f;
        }
    }

    auto attention_weights = softmax(*scores);

    return matmul(*attention_weights, value);
}

inline std::unique_ptr<Tensor> positional_encoding(int seq_len, int n_embd) {
    auto pe = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd});

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < n_embd; ++i) {
            if (i % 2 == 0) {
                pe->at({pos, i}) = std::sin(pos / std::pow(10000.0f, i / static_cast<float>(n_embd)));
            } else {
                pe->at({pos, i}) = std::cos(pos / std::pow(10000.0f, (i-1) / static_cast<float>(n_embd)));
            }
        }
    }

    return pe;
}

std::unique_ptr<Tensor> create_causal_mask(int seq_len);
std::unique_ptr<Tensor> create_padding_mask(const std::vector<int>& lengths, int max_len);
std::unique_ptr<Tensor> apply_mask_to_attention(const Tensor& attention_scores, const Tensor& mask);

std::unique_ptr<Tensor> gpt2_attention_forward(const Tensor& hidden_states,
                                               const Tensor& qkv_weight,
                                               const Tensor& qkv_bias,
                                               const Tensor& proj_weight,
                                               const Tensor& proj_bias,
                                               int n_head);

}