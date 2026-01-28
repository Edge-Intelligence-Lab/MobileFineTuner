#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace ops {

struct EmbeddingConfig {
    int vocab_size{50257};
    int embedding_dim{768};
    bool trainable{true};
    float dropout{0.0f};   // Unused placeholder for legacy configs
    int max_norm{-1};      // Clipping not implemented; placeholder
    bool sparse{false};    // Sparse updates not implemented; placeholder
};

class EmbeddingLayer {
public:
    explicit EmbeddingLayer(const EmbeddingConfig& config);
    ~EmbeddingLayer() = default;

    // Accepts token id list
    TensorPtr forward(const std::vector<int>& input_ids, bool training = false);
    // Accepts int tensor [S] or [B,S] (int32), returns [S,E] or [B,S,E]
    TensorPtr forward(const TensorPtr& input_ids, bool training = false);

    // Get/set parameters and gradients
    std::vector<TensorPtr> get_parameters() const;
    std::vector<TensorPtr> get_gradients() const;
    void zero_grad();
    // Simple parameter update (add externally computed updates to weights)
    void update_parameters(const std::vector<TensorPtr>& updates);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    size_t get_param_count() const;
    void set_weights(const TensorPtr& weights);
    TensorPtr get_embedding_weights() const { return weight_; }

private:
    EmbeddingConfig config_;
    TensorPtr weight_;       // [vocab_size, embedding_dim]
    TensorPtr grad_weight_;  // Same shape as weight_

    // Optional: keep small amount of state for compatibility with old impl
    std::vector<int> last_input_ids_;
    TensorPtr last_output_;
};

class PositionalEmbeddingLayer {
public:
    PositionalEmbeddingLayer(int max_position, int embedding_dim, bool trainable = true);
    ~PositionalEmbeddingLayer() = default;

    // Infer sequence length from input
    TensorPtr forward(const TensorPtr& input, bool training = false);
    // Or directly specify sequence length
    TensorPtr forward(int sequence_length, bool training = false);

    std::vector<TensorPtr> get_parameters() const;
    std::vector<TensorPtr> get_gradients() const;
    void zero_grad();
    void update_parameters(const std::vector<TensorPtr>& updates);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    size_t get_param_count() const;

private:
    int max_position_;
    int embedding_dim_;
    bool trainable_;

    TensorPtr position_embeddings_;        // [max_position, embedding_dim]
    TensorPtr grad_position_embeddings_;   // Same shape

    int last_seq_len_{0};
    TensorPtr last_output_;
};

}
