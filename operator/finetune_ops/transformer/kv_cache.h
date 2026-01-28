/**
 * @file kv_cache.h
 * @brief Key-Value cache implementsation for transforer inference
 * 
 * This file providess the KVCache structure for caching attention
 * key and value tensors during autoregressive generation. This
 * significantly speeds up inference by avoiding recomputation.
 */

#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

namespace ops {

/**
 * @brief Key-Value cache for transforer attention layers
 * 
 * Stores cached key and value tensors for each attention layer
 * to accelerate autoregressive generation. The cache maintains
 * a sliding window of past key-value pairs.
 */
struct KVCache {
    std::vector<std::unique_ptr<Tensor>> keys;    /**< Cached key tensors for each layer */
    std::vector<std::unique_ptr<Tensor>> values;  /**< Cached value tensors for each layer */
    int max_length;      /**< Maximum sequence length to cache */
    int current_length;  /**< Global cached tokens (deprecated, use layer_lengths) */
    std::vector<int> layer_lengths; /**< Per-layer cached token lengths */

    /**
     * @brief Construct KV cache with specified maximum length
     * @param max_len Maximum sequence length to cache (default: 2048)
     */
    KVCache(int max_len = 2048) : max_length(max_len), current_length(0) {}

    /**
     * @brief Initialize cache for a transforer model
     * @param n_layers Number of transforer layers
     * @param n_head Number of attention heads per layer
     * @param n_embd Embedding dimension
     */
    void initialize(int n_layers, int n_head, int n_embd);

    /**
     * @brief Update cache with new key-value pairs
     * @param new_keys New key tensors for each layer
     * @param new_values New value tensors for each layer
     */
    void update(const std::vector<std::unique_ptr<Tensor>>& new_keys,
                const std::vector<std::unique_ptr<Tensor>>& new_values);

    std::vector<std::unique_ptr<Tensor>> get_keys() const;
    std::vector<std::unique_ptr<Tensor>> get_values() const;

    // [Translated comment removed - see documentation]
    void update_layer(int layer_idx, const Tensor& new_key, const Tensor& new_value);
    std::unique_ptr<Tensor> get_key_layer(int layer_idx) const;
    std::unique_ptr<Tensor> get_value_layer(int layer_idx) const;

    void clear();

    bool is_full() const { return current_length >= max_length; }

    float usage_ratio() const { return static_cast<float>(current_length) / max_length; }
};

class CachedAttention {
private:
    int n_head_;
    int n_embd_;
    int head_dim_;

    std::unique_ptr<Tensor> qkv_weight_;
    std::unique_ptr<Tensor> qkv_bias_;
    std::unique_ptr<Tensor> proj_weight_;
    std::unique_ptr<Tensor> proj_bias_;

public:
    CachedAttention(int n_head, int n_embd);
    ~CachedAttention() = default;

    std::unique_ptr<Tensor> forward(const Tensor& hidden_states,
                                   KVCache& kv_cache,
                                   int layer_idx);

    std::unique_ptr<Tensor> compute_attention_scores(const Tensor& query,
                                                    const Tensor& key);

    std::unique_ptr<Tensor> apply_causal_mask(const Tensor& attention_scores);

    void print_info() const;
};

}