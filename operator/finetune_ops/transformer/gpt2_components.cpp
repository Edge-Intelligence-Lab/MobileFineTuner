#include "gpt2_components.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ops {

LayerNorm::LayerNorm(int n_embd, float epsilon) : epsilon_(epsilon) {
    weight_ = std::make_unique<Tensor>(std::vector<int>{n_embd});
    bias_ = std::make_unique<Tensor>(std::vector<int>{n_embd});

    for (int i = 0; i < n_embd; ++i) {
        weight_->data[i] = 1.0f;
        bias_->data[i] = 0.0f;
    }
}

std::unique_ptr<Tensor> LayerNorm::forward(const Tensor& input) {
    int batch_size = input.shape[0];
    int n_embd = input.shape[1];

    auto output = std::make_unique<Tensor>(input.shape);

    for (int b = 0; b < batch_size; ++b) {

        float mean = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            mean += input.at({b, i});
        }
        mean /= n_embd;

        float var = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            float diff = input.at({b, i}) - mean;
            var += diff * diff;
        }
        var /= n_embd;

        float std_dev = std::sqrt(var + epsilon_);
        for (int i = 0; i < n_embd; ++i) {
            float normalized = (input.at({b, i}) - mean) / std_dev;
            output->at({b, i}) = normalized * weight_->data[i] + bias_->data[i];
        }
    }

    return output;
}

void LayerNorm::print_info() const {
    std::cout << "LayerNorm Info:" << std::endl;
    std::cout << "  embedding[Output]: " << weight_->shape[0] << std::endl;
    std::cout << "  Epsilon: " << epsilon_ << std::endl;
}

FeedForward::FeedForward(int n_embd, int n_inner) {
    fc1_weight_ = std::make_unique<Tensor>(std::vector<int>{n_embd, n_inner});
    fc1_bias_ = std::make_unique<Tensor>(std::vector<int>{n_inner});
    fc2_weight_ = std::make_unique<Tensor>(std::vector<int>{n_inner, n_embd});
    fc2_bias_ = std::make_unique<Tensor>(std::vector<int>{n_embd});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : fc1_weight_->data) val = dist(gen);
    for (float& val : fc2_weight_->data) val = dist(gen);

    for (float& val : fc1_bias_->data) val = 0.0f;
    for (float& val : fc2_bias_->data) val = 0.0f;
}

std::unique_ptr<Tensor> FeedForward::forward(const Tensor& input) {

    auto hidden = linear(input, *fc1_weight_, *fc1_bias_);

    for (float& val : hidden->data) {
        val = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
    }

    auto output = linear(*hidden, *fc2_weight_, *fc2_bias_);

    return output;
}

void FeedForward::print_info() const {
    std::cout << "FeedForward Info:" << std::endl;
    std::cout << "  [Output]: " << fc1_weight_->shape[0] << std::endl;
    std::cout << "  [Output]: " << fc1_weight_->shape[1] << std::endl;
    std::cout << "  [Output]: " << fc2_weight_->shape[1] << std::endl;
}

Embedding::Embedding(int vocab_size, int n_embd) : vocab_size_(vocab_size), n_embd_(n_embd) {
    weight_ = std::make_unique<Tensor>(std::vector<int>{vocab_size, n_embd});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : weight_->data) {
        val = dist(gen);
    }
}

std::unique_ptr<Tensor> Embedding::forward(const std::vector<int>& input_ids) {
    int seq_len = static_cast<int>(input_ids.size());
    auto output = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd_});

    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];
        if (token_id >= 0 && token_id < vocab_size_) {
            for (int j = 0; j < n_embd_; ++j) {
                output->at({i, j}) = weight_->at({token_id, j});
            }
        } else {

            for (int j = 0; j < n_embd_; ++j) {
                output->at({i, j}) = 0.0f;
            }
        }
    }

    return output;
}

void Embedding::print_info() const {
    std::cout << "Embedding Info:" << std::endl;
    std::cout << "  [Output]size: " << vocab_size_ << std::endl;
    std::cout << "  embedding[Output]: " << n_embd_ << std::endl;
}

TransforerBlock::TransforerBlock(const GPT2Config& config)
    : n_head_(config.n_head), n_embd_(config.n_embd) {

    ln1_ = std::make_unique<LayerNorm>(config.n_embd, config.layer_norm_epsilon);
    ln2_ = std::make_unique<LayerNorm>(config.n_embd, config.layer_norm_epsilon);
    mlp_ = std::make_unique<FeedForward>(config.n_embd, config.n_inner);

    attn_qkv_weight_ = std::make_unique<Tensor>(std::vector<int>{config.n_embd, 3 * config.n_embd});
    attn_qkv_bias_ = std::make_unique<Tensor>(std::vector<int>{3 * config.n_embd});
    attn_proj_weight_ = std::make_unique<Tensor>(std::vector<int>{config.n_embd, config.n_embd});
    attn_proj_bias_ = std::make_unique<Tensor>(std::vector<int>{config.n_embd});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : attn_qkv_weight_->data) val = dist(gen);
    for (float& val : attn_proj_weight_->data) val = dist(gen);

    for (float& val : attn_qkv_bias_->data) val = 0.0f;
    for (float& val : attn_proj_bias_->data) val = 0.0f;
}

std::unique_ptr<Tensor> TransforerBlock::forward(const Tensor& hidden_states) {

    auto attn_output = gpt2_attention_forward(hidden_states,
                                             *attn_qkv_weight_, *attn_qkv_bias_,
                                             *attn_proj_weight_, *attn_proj_bias_,
                                             n_head_);

    auto residual1 = add(hidden_states, *attn_output);

    auto norm1 = ln1_->forward(*residual1);

    auto ff_output = mlp_->forward(*norm1);

    auto residual2 = add(*norm1, *ff_output);

    auto norm2 = ln2_->forward(*residual2);

    return norm2;
}

void TransforerBlock::print_info() const {
    std::cout << "TransforerBlock Info:" << std::endl;
    std::cout << "  attention[Output]: " << n_head_ << std::endl;
    std::cout << "  embedding[Output]: " << n_embd_ << std::endl;
    ln1_->print_info();
    ln2_->print_info();
    mlp_->print_info();
}

}