/**
 * @file layers.h
 * @brief Common neural network layers implementsation
 * 
 * This file providess implementsations of standard neural network layers
 * including Linear, ReLU, Dropout, LayerNorm, and others. All layers
 * inherit from the Module base class and support automatic differentiation.
 */

#pragma once

#include "module.h"
#include "../core/ops.h"
#include <cmath>
#include <cassert>

namespace ops {
namespace nn {

/**
 * @brief Linear (fully connected) layer
 * 
 * Applies a linear transforation to the input: y = xW^T + b
 * where W is the weight matrix and b is the optional bias vector.
 */
class Linear : public Module {
public:
    /**
     * @brief Construct a linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param bias Whether to include bias term (default: true)
     */
    Linear(int in_features, int out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {

        weight_ = create_parameter({out_features, in_features}, true, "xavier_unifor");
        register_parameter("weight", weight_);

        if (has_bias_) {
            bias_ = create_parameter({out_features}, true, "zeros");
            register_parameter("bias", bias_);
        }
    }

    TensorPtr forward(const TensorPtr& input) override {
        if (has_bias_) {
            return ops::linear(input, weight_->data(), bias_->data());
        } else {
            return ops::linear(input, weight_->data());
        }
    }

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

protected:
    std::string get_module_name() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + std::to_string(out_features_) + ")";
    }

private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    ParameterPtr weight_;
    ParameterPtr bias_;
};

class Embedding : public Module {
public:
    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim), padding_idx_(padding_idx) {

        weight_ = create_parameter({num_embeddings, embedding_dim}, true, "normal");
        register_parameter("weight", weight_);

        if (padding_idx_ >= 0) {

        }
    }

    TensorPtr forward(const TensorPtr& input) override {

        return ops::embedding_lookup(weight_->data(), input, padding_idx_);
    }

    int num_embeddings() const { return num_embeddings_; }
    int embedding_dim() const { return embedding_dim_; }

protected:
    std::string get_module_name() const override {
        return "Embedding(" + std::to_string(num_embeddings_) + ", " + std::to_string(embedding_dim_) + ")";
    }

private:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    ParameterPtr weight_;
};

class LayerNorm : public Module {
public:
    LayerNorm(const std::vector<int64_t>& normalized_shape, float eps = 1e-5, bool elementwise_affine = true)
        : normalized_shape_(normalized_shape), eps_(eps), elementwise_affine_(elementwise_affine) {

        if (elementwise_affine_) {

            weight_ = create_parameter(normalized_shape, true, "ones");
            bias_ = create_parameter(normalized_shape, true, "zeros");
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }

    LayerNorm(int normalized_shape, float eps = 1e-5, bool elementwise_affine = true)
        : LayerNorm({normalized_shape}, eps, elementwise_affine) {}

    TensorPtr forward(const TensorPtr& input) override {
        if (elementwise_affine_) {
            return functional::layer_norm(input, normalized_shape_, weight_->data(), bias_->data(), eps_);
        } else {
            return functional::layer_norm(input, normalized_shape_, nullptr, nullptr, eps_);
        }
    }

protected:
    std::string get_module_name() const override {
        return "LayerNorm(" + std::to_string(normalized_shape_[0]) + ")";
    }

private:
    std::vector<int64_t> normalized_shape_;
    float eps_;
    bool elementwise_affine_;
    ParameterPtr weight_;
    ParameterPtr bias_;
};

class Dropout : public Module {
public:
    Dropout(float p = 0.5) : p_(p) {}

    TensorPtr forward(const TensorPtr& input) override {
        return functional::dropout(input, p_, training());
    }

    float p() const { return p_; }

protected:
    std::string get_module_name() const override {
        return "Dropout(p=" + std::to_string(p_) + ")";
    }

private:
    float p_;
};

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int d_model, int num_heads, float dropout = 0.0, bool bias = true)
        : d_model_(d_model), num_heads_(num_heads), dropout_(dropout) {

        assert(d_model % num_heads == 0);
        head_dim_ = d_model / num_heads;

        q_linear_ = std::make_shared<Linear>(d_model, d_model, bias);
        k_linear_ = std::make_shared<Linear>(d_model, d_model, bias);
        v_linear_ = std::make_shared<Linear>(d_model, d_model, bias);
        out_linear_ = std::make_shared<Linear>(d_model, d_model, bias);

        register_module("q_linear", q_linear_);
        register_module("k_linear", k_linear_);
        register_module("v_linear", v_linear_);
        register_module("out_linear", out_linear_);

        if (dropout > 0.0) {
            dropout_layer_ = std::make_shared<Dropout>(dropout);
            register_module("dropout", dropout_layer_);
        }
    }

    TensorPtr forward(const TensorPtr& input) override {
        return forward(input, input, input);
    }

    TensorPtr forward(const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
                     const TensorPtr& attn_mask = nullptr) {

        auto batch_size = query->size(0);
        auto seq_len = query->size(1);

        auto Q = q_linear_->forward(query);
        auto K = k_linear_->forward(key);
        auto V = v_linear_->forward(value);

        Q = Q->reshape({batch_size, seq_len, num_heads_, head_dim_});
        K = K->reshape({batch_size, seq_len, num_heads_, head_dim_});
        V = V->reshape({batch_size, seq_len, num_heads_, head_dim_});

        Q = Q->transpose(1, 2);
        K = K->transpose(1, 2);
        V = V->transpose(1, 2);

        auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        auto scores = functional::matmul(Q, K->transpose(-2, -1));
        scores = functional::mul(scores, scale);

        if (attn_mask) {
            scores = functional::add(scores, attn_mask);
        }

        auto attn_weights = functional::softmax(scores, -1);

        if (dropout_layer_) {
            attn_weights = dropout_layer_->forward(attn_weights);
        }

        auto context = functional::matmul(attn_weights, V);

        context = context->transpose(1, 2);
        context = context->reshape({batch_size, seq_len, d_model_});

        auto output = out_linear_->forward(context);

        return output;
    }

    int d_model() const { return d_model_; }
    int num_heads() const { return num_heads_; }

protected:
    std::string get_module_name() const override {
        return "MultiHeadAttention(d_model=" + std::to_string(d_model_) +
               ", num_heads=" + std::to_string(num_heads_) + ")";
    }

private:
    int d_model_;
    int num_heads_;
    int head_dim_;
    float dropout_;

    std::shared_ptr<Linear> q_linear_;
    std::shared_ptr<Linear> k_linear_;
    std::shared_ptr<Linear> v_linear_;
    std::shared_ptr<Linear> out_linear_;
    std::shared_ptr<Dropout> dropout_layer_;
};

class FeedForward : public Module {
public:
    FeedForward(int d_model, int d_ff, float dropout = 0.0, const std::string& activation = "relu")
        : d_model_(d_model), d_ff_(d_ff), activation_(activation) {

        linear1_ = std::make_shared<Linear>(d_model, d_ff);
        linear2_ = std::make_shared<Linear>(d_ff, d_model);

        register_module("linear1", linear1_);
        register_module("linear2", linear2_);

        if (dropout > 0.0) {
            dropout_layer_ = std::make_shared<Dropout>(dropout);
            register_module("dropout", dropout_layer_);
        }
    }

    TensorPtr forward(const TensorPtr& input) override {
        auto x = linear1_->forward(input);

        if (activation_ == "relu") {
            x = functional::relu(x);
        } else if (activation_ == "gelu") {
            x = functional::gelu(x);
        } else if (activation_ == "sigmoid") {
            x = functional::sigmoid(x);
        }

        if (dropout_layer_) {
            x = dropout_layer_->forward(x);
        }

        x = linear2_->forward(x);
        return x;
    }

protected:
    std::string get_module_name() const override {
        return "FeedForward(" + std::to_string(d_model_) + " -> " +
               std::to_string(d_ff_) + " -> " + std::to_string(d_model_) + ")";
    }

private:
    int d_model_;
    int d_ff_;
    std::string activation_;

    std::shared_ptr<Linear> linear1_;
    std::shared_ptr<Linear> linear2_;
    std::shared_ptr<Dropout> dropout_layer_;
};

class TransforerBlock : public Module {
public:
    TransforerBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1,
                    const std::string& activation = "gelu", bool pre_norm = true)
        : d_model_(d_model), pre_norm_(pre_norm) {

        self_attn_ = std::make_shared<MultiHeadAttention>(d_model, num_heads, dropout);
        register_module("self_attn", self_attn_);

        feed_forward_ = std::make_shared<FeedForward>(d_model, d_ff, dropout, activation);
        register_module("feed_forward", feed_forward_);

        norm1_ = std::make_shared<LayerNorm>(d_model);
        norm2_ = std::make_shared<LayerNorm>(d_model);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);

        if (dropout > 0.0) {
            dropout1_ = std::make_shared<Dropout>(dropout);
            dropout2_ = std::make_shared<Dropout>(dropout);
            register_module("dropout1", dropout1_);
            register_module("dropout2", dropout2_);
        }
    }

    TensorPtr forward(const TensorPtr& input) override {
        return forward(input, nullptr);
    }

    TensorPtr forward(const TensorPtr& input, const TensorPtr& attn_mask) {
        TensorPtr x = input;

        if (pre_norm_) {

            auto norm_x = norm1_->forward(x);
            auto attn_out = self_attn_->forward(norm_x, norm_x, norm_x, attn_mask);
            if (dropout1_) attn_out = dropout1_->forward(attn_out);
            x = functional::add(x, attn_out);

            norm_x = norm2_->forward(x);
            auto ff_out = feed_forward_->forward(norm_x);
            if (dropout2_) ff_out = dropout2_->forward(ff_out);
            x = functional::add(x, ff_out);
        } else {

            auto attn_out = self_attn_->forward(x, x, x, attn_mask);
            if (dropout1_) attn_out = dropout1_->forward(attn_out);
            x = norm1_->forward(functional::add(x, attn_out));

            auto ff_out = feed_forward_->forward(x);
            if (dropout2_) ff_out = dropout2_->forward(ff_out);
            x = norm2_->forward(functional::add(x, ff_out));
        }

        return x;
    }

protected:
    std::string get_module_name() const override {
        return "TransforerBlock(d_model=" + std::to_string(d_model_) + ")";
    }

private:
    int d_model_;
    bool pre_norm_;

    std::shared_ptr<MultiHeadAttention> self_attn_;
    std::shared_ptr<FeedForward> feed_forward_;
    std::shared_ptr<LayerNorm> norm1_;
    std::shared_ptr<LayerNorm> norm2_;
    std::shared_ptr<Dropout> dropout1_;
    std::shared_ptr<Dropout> dropout2_;
};

}
}
