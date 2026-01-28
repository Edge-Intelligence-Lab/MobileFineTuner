#include "kv_cache.h"
#include "basic_ops.h"
#include <iostream>
#include <algorithm>
#include <random>

namespace ops {

void KVCache::initialize(int n_layers, int n_head, int n_embd) {
    keys.clear();
    values.clear();
    current_length = 0;

    keys.reserve(n_layers);
    values.reserve(n_layers);
    for (int i = 0; i < n_layers; i++) {
        keys.push_back(nullptr);
        values.push_back(nullptr);
    }
}

void KVCache::update(const std::vector<std::unique_ptr<Tensor>>& new_keys,
                     const std::vector<std::unique_ptr<Tensor>>& new_values) {
    if (new_keys.size() != new_values.size()) {
        std::cerr << "KV Cacheupdate[Output]：keyandvalue[Output]match" << std::endl;
        return;
    }
    for (size_t i = 0; i < new_keys.size(); i++) {
        if (keys[i] == nullptr) {

            int dim = new_keys[i]->shape[1];
            keys[i] = std::make_unique<Tensor>(std::vector<int>{max_length, dim});
            values[i] = std::make_unique<Tensor>(std::vector<int>{max_length, dim});
        }
        int dim = new_keys[i]->shape[1];
        if (current_length < max_length) {
            for (int j = 0; j < dim; j++) {

                keys[i]->at({current_length, j}) = new_keys[i]->at({0, j});
                values[i]->at({current_length, j}) = new_values[i]->at({0, j});
            }
        }
    }
    current_length++;
}

std::vector<std::unique_ptr<Tensor>> KVCache::get_keys() const {
    std::vector<std::unique_ptr<Tensor>> result;
    for (const auto& key : keys) {
        if (key != nullptr && current_length > 0) {
            int dim = key->shape[1];
            auto key_slice = std::make_unique<Tensor>(std::vector<int>{current_length, dim});
            for (int i = 0; i < current_length; i++) {
                for (int j = 0; j < dim; j++) {
                    key_slice->at({i, j}) = key->at({i, j});
                }
            }
            result.push_back(std::move(key_slice));
        } else {
            result.push_back(nullptr);
        }
    }
    return result;
}

std::vector<std::unique_ptr<Tensor>> KVCache::get_values() const {
    std::vector<std::unique_ptr<Tensor>> result;
    for (const auto& value : values) {
        if (value != nullptr && current_length > 0) {
            int dim = value->shape[1];
            auto value_slice = std::make_unique<Tensor>(std::vector<int>{current_length, dim});
            for (int i = 0; i < current_length; i++) {
                for (int j = 0; j < dim; j++) {
                    value_slice->at({i, j}) = value->at({i, j});
                }
            }
            result.push_back(std::move(value_slice));
        } else {
            result.push_back(nullptr);
        }
    }
    return result;
}

void KVCache::clear() {
    current_length = 0;
}

void KVCache::update_layer(int layer_idx, const Tensor& new_key, const Tensor& new_value) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(keys.size())) {
        std::cerr << "KVCache::update_layer: invalid layer index" << std::endl;
        return;
    }
    int dim = new_key.shape[1];
    if (!keys[layer_idx]) {
        keys[layer_idx] = std::make_unique<Tensor>(std::vector<int>{max_length, dim});
        values[layer_idx] = std::make_unique<Tensor>(std::vector<int>{max_length, dim});
    }
    if (current_length < max_length) {
        for (int j = 0; j < dim; ++j) {
            keys[layer_idx]->at({current_length, j}) = new_key.at({0, j});
            values[layer_idx]->at({current_length, j}) = new_value.at({0, j});
        }
        current_length++;
    }
}

std::unique_ptr<Tensor> KVCache::get_key_layer(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(keys.size())) {
        return nullptr;
    }
    const auto& key = keys[layer_idx];
    if (!key || current_length == 0) return nullptr;
    int dim = key->shape[1];
    auto key_slice = std::make_unique<Tensor>(std::vector<int>{current_length, dim});
    for (int i = 0; i < current_length; ++i) {
        for (int j = 0; j < dim; ++j) {
            key_slice->at({i, j}) = key->at({i, j});
        }
    }
    return key_slice;
}

std::unique_ptr<Tensor> KVCache::get_value_layer(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(values.size())) {
        return nullptr;
    }
    const auto& value = values[layer_idx];
    if (!value || current_length == 0) return nullptr;
    int dim = value->shape[1];
    auto value_slice = std::make_unique<Tensor>(std::vector<int>{current_length, dim});
    for (int i = 0; i < current_length; ++i) {
        for (int j = 0; j < dim; ++j) {
            value_slice->at({i, j}) = value->at({i, j});
        }
    }
    return value_slice;
}

std::unique_ptr<Tensor> slice_rows(const Tensor& input, int col_start, int col_end) {

    int rows = input.shape[0];
    int cols = col_end - col_start;
    auto result = std::make_unique<Tensor>(std::vector<int>{rows, cols});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result->at({i, j}) = input.at({i, col_start + j});
        }
    }
    return result;
}

std::unique_ptr<Tensor> scale_tensor(const Tensor& input, float scale) {
    auto result = std::make_unique<Tensor>(input.shape);
    for (size_t i = 0; i < input.data.size(); ++i) {
        result->data[i] = input.data[i] * scale;
    }
    return result;
}

CachedAttention::CachedAttention(int n_head, int n_embd)
    : n_head_(n_head), n_embd_(n_embd), head_dim_(n_embd / n_head) {

    qkv_weight_ = std::make_unique<Tensor>(std::vector<int>{n_embd, n_embd * 3});
    qkv_bias_ = std::make_unique<Tensor>(std::vector<int>{1, n_embd * 3});
    proj_weight_ = std::make_unique<Tensor>(std::vector<int>{n_embd, n_embd});
    proj_bias_ = std::make_unique<Tensor>(std::vector<int>{1, n_embd});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& v : qkv_weight_->data) v = dist(gen);
    for (auto& v : qkv_bias_->data) v = dist(gen);
    for (auto& v : proj_weight_->data) v = dist(gen);
    for (auto& v : proj_bias_->data) v = dist(gen);
}

std::unique_ptr<Tensor> CachedAttention::forward(const Tensor& hidden_states,
                                                 KVCache& kv_cache,
                                                 int layer_idx) {
    int seq_len = hidden_states.shape[0];

    auto qkv = ops::linear(hidden_states, *qkv_weight_, *qkv_bias_);

        // [Translated]
    auto qkv_reshaped = ops::reshape(*qkv, std::vector<int>{seq_len, n_head_ * head_dim_ * 3});
    auto q = slice_rows(*qkv_reshaped, 0, n_head_ * head_dim_);
    auto k = slice_rows(*qkv_reshaped, n_head_ * head_dim_, 2 * n_head_ * head_dim_);
    auto v = slice_rows(*qkv_reshaped, 2 * n_head_ * head_dim_, 3 * n_head_ * head_dim_);

    // usebylayerupdateinterface
    kv_cache.update_layer(layer_idx, *k, *v);

        // [Translated]
    auto cached_k = kv_cache.get_key_layer(layer_idx);
    auto cached_v = kv_cache.get_value_layer(layer_idx);
    if (cached_k) { k = std::move(cached_k); }
    if (cached_v) { v = std::move(cached_v); }

    auto attention_scores = compute_attention_scores(*q, *k);

    auto masked_scores = apply_causal_mask(*attention_scores);

    auto attention_weights = ops::softmax(*masked_scores);

    auto context = ops::matmul(*attention_weights, *v);

    auto context_reshaped = ops::reshape(*context, std::vector<int>{seq_len, n_embd_});

    auto output = ops::linear(*context_reshaped, *proj_weight_, *proj_bias_);
    return output;
}

std::unique_ptr<Tensor> CachedAttention::compute_attention_scores(const Tensor& query,
                                                                  const Tensor& key) {

    auto key_transposed = ops::transpose(key);
    auto scores = ops::matmul(query, *key_transposed);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto scaled_scores = scale_tensor(*scores, scale);
    return scaled_scores;
}

std::unique_ptr<Tensor> CachedAttention::apply_causal_mask(const Tensor& attention_scores) {
    int seq_len = attention_scores.shape[0];
    auto masked_scores = std::make_unique<Tensor>(attention_scores);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                masked_scores->at({i, j}) = -1e9f;
            }
        }
    }
    return masked_scores;
}

void CachedAttention::print_info() const {
    std::cout << "CachedAttention Info:" << std::endl;
    std::cout << "  n_head: " << n_head_ << std::endl;
    std::cout << "  n_embd: " << n_embd_ << std::endl;
    std::cout << "  head_dim: " << head_dim_ << std::endl;
}

}