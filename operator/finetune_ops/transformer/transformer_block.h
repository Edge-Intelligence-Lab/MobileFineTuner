/**
 * @file transforer_block.h
 * @brief Transforer block implementsation
 * 
 * This file providess the TransforerBlock class which implementss
 * a complete transforer block consisting of multi-head attention
 * and feed-forward networks with layer normalization and residual connections.
 */

#pragma once

#include "../layer/attention_layer.h"
#include "../layer/mlp_layer.h"
#include <memory>

namespace ops {

/**
 * @brief Configuration structure for transforer blocks
 * 
 * Contains all architectural parameters needed to configure
 * a transforer block including attention heads, MLP dimensions,
 * and normalization settings.
 */
struct TransforerBlockConfig {
    int hidden_size;  /**< Hidden dimension size */
    int num_heads;    /**< Number of attention heads */
    int mlp_dim;      /**< MLP intermediate dimension */
    float dropout;    /**< Dropout rate */
    bool pre_norm;    /**< Whether to use pre-normalization */

    /**
     * @brief Default transforer block configuration
     * 
     * Creates a configuration with default values:
     * - hidden_size: 0 (must be set)
     * - num_heads: 1
     * - mlp_dim: 0 (must be set)
     * - dropout: 0.1
     * - pre_norm: true
     */
    TransforerBlockConfig() : hidden_size(0), num_heads(1),
        mlp_dim(0), dropout(0.1f), pre_norm(true) {}

    /**
     * @brief Construct transforer block configuration
     * @param hidden_size_ Hidden dimension size
     * @param num_heads_ Number of attention heads
     * @param mlp_dim_ MLP intermediate dimension
     * @param dropout_ Dropout rate (default: 0.1)
     * @param pre_norm_ Whether to use pre-normalization (default: true)
     */
    TransforerBlockConfig(
        int hidden_size_,
        int num_heads_,
        int mlp_dim_,
        float dropout_ = 0.1f,
        bool pre_norm_ = true
    ) : hidden_size(hidden_size_),
        num_heads(num_heads_),
        mlp_dim(mlp_dim_),
        dropout(dropout_),
        pre_norm(pre_norm_) {}
};

class TransforerBlock {
public:
    TransforerBlock(const TransforerBlockConfig& config);

    TensorPtr forward(const Tensor& input, bool training = false);
    TensorPtr forward_with_cache(const Tensor& input,
                               const Tensor& key_cache,
                               const Tensor& value_cache,
                               bool training = false);

    TensorPtr backward(const Tensor& grad_output);

    std::vector<TensorPtr> get_parameters();
    std::vector<TensorPtr> get_gradients();

    void update_parameters(const std::vector<TensorPtr>& updates);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    size_t get_param_count() const;

private:
    TransforerBlockConfig config_;

    std::unique_ptr<AttentionLayer> self_attention_;
    std::unique_ptr<MLPLayer> mlp_;

    TensorPtr attention_ln_weight_;
    TensorPtr attention_ln_bias_;
    TensorPtr mlp_ln_weight_;
    TensorPtr mlp_ln_bias_;

    TensorPtr grad_attention_ln_weight_;
    TensorPtr grad_attention_ln_bias_;
    TensorPtr grad_mlp_ln_weight_;
    TensorPtr grad_mlp_ln_bias_;

    TensorPtr last_input_;
    TensorPtr last_attention_input_;
    TensorPtr last_attention_output_;
    TensorPtr last_mlp_input_;
    TensorPtr last_mlp_output_;

    TensorPtr layer_norm(const Tensor& input,
                        const Tensor& weight,
                        const Tensor& bias);
    TensorPtr layer_norm_backward(const Tensor& grad_output,
                                const Tensor& input,
                                const Tensor& weight,
                                const Tensor& bias);
};

}
