#include "attention.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ops {

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config) : config_(config) {
    qkv_weight_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, 3 * config_.n_embd});
    qkv_bias_ = std::make_unique<Tensor>(std::vector<int>{3 * config_.n_embd});
    proj_weight_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, config_.n_embd});
    proj_bias_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd});

    initialize_weights();
}

void MultiHeadAttention::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (float& val : qkv_weight_->data) val = dist(gen);
    for (float& val : proj_weight_->data) val = dist(gen);

    for (float& val : qkv_bias_->data) val = 0.0f;
    for (float& val : proj_bias_->data) val = 0.0f;
}

std::unique_ptr<Tensor> MultiHeadAttention::forward(const Tensor& hidden_states) {
    int seq_len = hidden_states.shape[0];
    int n_embd = hidden_states.shape[1];
    assert(n_embd == config_.n_embd);

    auto qkv = linear(hidden_states, *qkv_weight_, *qkv_bias_);

    auto q = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    auto k = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    auto v = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < config_.n_embd; ++j) {
            q->at({i, j}) = qkv->at({i, j});
            k->at({i, j}) = qkv->at({i, j + config_.n_embd});
            v->at({i, j}) = qkv->at({i, j + 2 * config_.n_embd});
        }
    }

    auto q_heads = reshape_for_heads(*q, seq_len);
    auto k_heads = reshape_for_heads(*k, seq_len);
    auto v_heads = reshape_for_heads(*v, seq_len);

    auto attn_output = scaled_dot_product_attention(*q_heads, *k_heads, *v_heads);

    auto merged = merge_heads(*attn_output, seq_len);

    auto output = linear(*merged, *proj_weight_, *proj_bias_);

    return output;
}

std::unique_ptr<Tensor> MultiHeadAttention::reshape_for_heads(const Tensor& input, int seq_len) const {

    auto result = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_head, config_.head_dim});

    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < config_.n_head; ++h) {
            for (int d = 0; d < config_.head_dim; ++d) {
                int input_idx = i * config_.n_embd + h * config_.head_dim + d;
                result->at({i, h, d}) = input.data[input_idx];
            }
        }
    }

    return result;
}

std::unique_ptr<Tensor> MultiHeadAttention::merge_heads(const Tensor& input, int seq_len) const {

    auto result = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});

    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < config_.n_head; ++h) {
            for (int d = 0; d < config_.head_dim; ++d) {
                int output_idx = i * config_.n_embd + h * config_.head_dim + d;
                result->data[output_idx] = input.at({i, h, d});
            }
        }
    }

    return result;
}

std::unique_ptr<Tensor> MultiHeadAttention::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v) const {
    int seq_len = q.shape[0];
    int n_head = q.shape[1];
    int head_dim = q.shape[2];
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto result = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, head_dim});

    for (int h = 0; h < n_head; ++h) {

        auto q_head = std::make_unique<Tensor>(std::vector<int>{seq_len, head_dim});
        auto k_head = std::make_unique<Tensor>(std::vector<int>{seq_len, head_dim});
        auto v_head = std::make_unique<Tensor>(std::vector<int>{seq_len, head_dim});

        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                q_head->at({i, d}) = q.at({i, h, d});
                k_head->at({i, d}) = k.at({i, h, d});
                v_head->at({i, d}) = v.at({i, h, d});
            }
        }

        auto scores = matmul(*q_head, *transpose(*k_head));

        for (float& val : scores->data) {
            val *= scale;
        }

        scores = apply_causal_mask(*scores);

        auto attention_weights = softmax(*scores);

        auto head_output = matmul(*attention_weights, *v_head);

        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                result->at({i, h, d}) = head_output->at({i, d});
            }
        }
    }

    return result;
}

std::unique_ptr<Tensor> MultiHeadAttention::apply_causal_mask(const Tensor& scores) const {
    auto masked_scores = clone(scores);

    int seq_len = scores.shape[0];

    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            masked_scores->at({i, j}) = -1e9f;
        }
    }

    return masked_scores;
}

void MultiHeadAttention::print_info() const {
    std::cout << "MultiHeadAttention Info:" << std::endl;
    std::cout << "  attention[Output]: " << config_.n_head << std::endl;
    std::cout << "  [Output]: " << config_.head_dim << std::endl;
    std::cout << "  embedding[Output]: " << config_.n_embd << std::endl;
    std::cout << "  QKV [Output]: " << config_.n_embd << " x " << 3 * config_.n_embd << std::endl;
    std::cout << "  project[Output]: " << config_.n_embd << " x " << config_.n_embd << std::endl;
}

std::unique_ptr<Tensor> create_causal_mask(int seq_len) {
    auto mask = std::make_unique<Tensor>(std::vector<int>{seq_len, seq_len});

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j <= i) {
                mask->at({i, j}) = 0.0f;
            } else {
                mask->at({i, j}) = -1e9f;
            }
        }
    }

    return mask;
}

std::unique_ptr<Tensor> create_padding_mask(const std::vector<int>& lengths, int max_len) {
    auto mask = std::make_unique<Tensor>(std::vector<int>{max_len, max_len});

    for (int i = 0; i < max_len; ++i) {
        for (int j = 0; j < max_len; ++j) {
            if (i < lengths.size() && j < lengths[i]) {
                mask->at({i, j}) = 0.0f;
            } else {
                mask->at({i, j}) = -1e9f;
            }
        }
    }

    return mask;
}

std::unique_ptr<Tensor> apply_mask_to_attention(const Tensor& attention_scores, const Tensor& mask) {
    assert(attention_scores.shape == mask.shape);

    auto result = clone(attention_scores);

    for (int i = 0; i < static_cast<int>(result->data.size()); ++i) {
        result->data[i] += mask.data[i];
    }

    return result;
}

std::unique_ptr<Tensor> gpt2_attention_forward(const Tensor& hidden_states,
                                               const Tensor& qkv_weight,
                                               const Tensor& qkv_bias,
                                               const Tensor& proj_weight,
                                               const Tensor& proj_bias,
                                               int n_head) {
    int seq_len = hidden_states.shape[0];
    int n_embd = hidden_states.shape[1];
    int head_dim = n_embd / n_head;

    auto qkv = linear(hidden_states, qkv_weight, qkv_bias);

    auto q = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd});
    auto k = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd});
    auto v = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd});

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            q->at({i, j}) = qkv->at({i, j});
            k->at({i, j}) = qkv->at({i, j + n_embd});
            v->at({i, j}) = qkv->at({i, j + 2 * n_embd});
        }
    }

    auto q_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, head_dim});
    auto k_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, head_dim});
    auto v_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, head_dim});

    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                q_heads->at({i, h, d}) = q->at({i, h * head_dim + d});
                k_heads->at({i, h, d}) = k->at({i, h * head_dim + d});
                v_heads->at({i, h, d}) = v->at({i, h * head_dim + d});
            }
        }
    }

    auto scores = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, seq_len});
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_heads->at({i, h, d}) * k_heads->at({j, h, d});
                }
                scores->at({i, h, j}) = score * scale;
            }
        }
    }

    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                scores->at({i, h, j}) = -1e9f;
            }
        }
    }

    auto attention_weights = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, seq_len});
    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < seq_len; ++i) {

            float max_score = scores->at({i, h, 0});
            for (int j = 1; j < seq_len; ++j) {
                max_score = std::max(max_score, scores->at({i, h, j}));
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float exp_val = std::exp(scores->at({i, h, j}) - max_score);
                attention_weights->at({i, h, j}) = exp_val;
                sum_exp += exp_val;
            }

            for (int j = 0; j < seq_len; ++j) {
                attention_weights->at({i, h, j}) /= sum_exp;
            }
        }
    }

    auto attn_output = std::make_unique<Tensor>(std::vector<int>{seq_len, n_head, head_dim});
    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float output_val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    output_val += attention_weights->at({i, h, j}) * v_heads->at({j, h, d});
                }
                attn_output->at({i, h, d}) = output_val;
            }
        }
    }

    auto merged = std::make_unique<Tensor>(std::vector<int>{seq_len, n_embd});
    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                merged->at({i, h * head_dim + d}) = attn_output->at({i, h, d});
            }
        }
    }

    auto output = linear(*merged, proj_weight, proj_bias);

    return output;
}

}