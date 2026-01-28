#include "embedding.h"
#include "../core/ops.h"
#include <random>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace ops {

EmbeddingLayer::EmbeddingLayer(const EmbeddingConfig& config)
    : config_(config) {
    weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(config_.vocab_size),
                             static_cast<int64_t>(config_.embedding_dim)},
        kFloat32, kCPU);
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / (config_.vocab_size + config_.embedding_dim));
    std::normal_distribution<float> dist(0.0f, std_dev);
    float* w = weight_->data<float>();
    for (int64_t i = 0; i < weight_->numel(); ++i) {
        w[i] = dist(gen);
    }
    if (config_.trainable) {
        grad_weight_ = zeros(weight_->shape(), kFloat32, kCPU);
    }
}

TensorPtr EmbeddingLayer::forward(const std::vector<int>& input_ids, bool training) {
    int64_t S = static_cast<int64_t>(input_ids.size());
    auto ids = zeros({S}, kInt32, kCPU);
    int32_t* id_data = ids->data<int32_t>();
    for (int64_t i = 0; i < S; ++i) id_data[i] = static_cast<int32_t>(input_ids[static_cast<size_t>(i)]);
    return forward(ids, training);
}

TensorPtr EmbeddingLayer::forward(const TensorPtr& input_ids, bool training) {
    if (!input_ids) throw std::runtime_error("EmbeddingLayer::forward: input_ids is null");
    if (input_ids->dtype() != kInt32) {
        throw std::runtime_error("EmbeddingLayer::forward expects int32 input_ids");
    }
    const auto& ishape = input_ids->shape();
    if (ishape.size() != 1 && ishape.size() != 2) {
        throw std::runtime_error("EmbeddingLayer::forward expects [S] or [B,S] int32");
    }
    const int64_t E = config_.embedding_dim;
    const int64_t V = config_.vocab_size;
    const int32_t* ids = input_ids->data<int32_t>();

    TensorPtr output;
    if (ishape.size() == 1) {
        int64_t S = ishape[0];
        output = zeros({S, E}, kFloat32, kCPU);
        float* out = output->data<float>();
        const float* W = weight_->data<float>();
        for (int64_t s = 0; s < S; ++s) {
            int32_t idx = ids[s];
            if (idx >= 0 && idx < V) {
                std::memcpy(out + s * E, W + static_cast<int64_t>(idx) * E, sizeof(float) * E);
            } else {
                std::memset(out + s * E, 0, sizeof(float) * E);
            }
        }
    } else {
        int64_t B = ishape[0];
        int64_t S = ishape[1];
        output = zeros({B, S, E}, kFloat32, kCPU);
        float* out = output->data<float>();
        const float* W = weight_->data<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S; ++s) {
                int32_t idx = ids[b * S + s];
                float* dst = out + (b * S + s) * E;
                if (idx >= 0 && idx < V) {
                    std::memcpy(dst, W + static_cast<int64_t>(idx) * E, sizeof(float) * E);
                } else {
                    std::memset(dst, 0, sizeof(float) * E);
                }
            }
        }
    }

    if (training && config_.trainable && ishape.size() == 1) {
        last_input_ids_.assign(input_ids->data<int32_t>(), input_ids->data<int32_t>() + ishape[0]);
        last_output_ = output;
    }
    return output;
}

std::vector<TensorPtr> EmbeddingLayer::get_parameters() const {
    return {weight_};
}

std::vector<TensorPtr> EmbeddingLayer::get_gradients() const {
    if (config_.trainable && grad_weight_) return {grad_weight_};
    return {};
}

void EmbeddingLayer::zero_grad() {
    if (grad_weight_) {
        std::memset(grad_weight_->data_ptr(), 0, static_cast<size_t>(grad_weight_->numel() * sizeof(float)));
    }
}

void EmbeddingLayer::update_parameters(const std::vector<TensorPtr>& updates) {
    if (!config_.trainable || updates.empty() || !updates[0]) return;
    if (updates[0]->shape() != weight_->shape()) {
        throw std::runtime_error("EmbeddingLayer::update_parameters: shape mismatch");
    }
    float* w = weight_->data<float>();
    const float* u = updates[0]->data<float>();
    for (int64_t i = 0; i < weight_->numel(); ++i) {
        w[i] += u[i];
    }
}

void EmbeddingLayer::save_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    int vs = config_.vocab_size;
    int ed = config_.embedding_dim;
    f.write(reinterpret_cast<const char*>(&vs), sizeof(int));
    f.write(reinterpret_cast<const char*>(&ed), sizeof(int));
    const float* w = weight_->data<float>();
    f.write(reinterpret_cast<const char*>(w), static_cast<std::streamsize>(weight_->numel() * sizeof(float)));
}

void EmbeddingLayer::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);
    int vs, ed;
    f.read(reinterpret_cast<char*>(&vs), sizeof(int));
    f.read(reinterpret_cast<char*>(&ed), sizeof(int));
    if (vs != config_.vocab_size || ed != config_.embedding_dim) {
        throw std::runtime_error("EmbeddingLayer::load_weights: config mismatch");
    }
    float* w = weight_->data<float>();
    f.read(reinterpret_cast<char*>(w), static_cast<std::streamsize>(weight_->numel() * sizeof(float)));
}

size_t EmbeddingLayer::get_param_count() const {
    return static_cast<size_t>(weight_->numel());
}

void EmbeddingLayer::set_weights(const TensorPtr& weights) {
    if (!weights) throw std::runtime_error("EmbeddingLayer::set_weights: null");
    if (weights->shape() != weight_->shape()) {
        throw std::runtime_error("EmbeddingLayer::set_weights: shape mismatch");
    }
    weight_ = weights;
}

PositionalEmbeddingLayer::PositionalEmbeddingLayer(int max_position, int embedding_dim, bool trainable)
    : max_position_(max_position),
      embedding_dim_(embedding_dim),
      trainable_(trainable) {
    position_embeddings_ = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(max_position_), static_cast<int64_t>(embedding_dim_)},
        kFloat32, kCPU);
    float* pos = position_embeddings_->data<float>();
    for (int p = 0; p < max_position_; ++p) {
        for (int i = 0; i < embedding_dim_; ++i) {
            float angle = static_cast<float>(p) / std::pow(10000.0f, 2.0f * (i / 2) / static_cast<float>(embedding_dim_));
            pos[p * embedding_dim_ + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    if (trainable_) {
        grad_position_embeddings_ = zeros(position_embeddings_->shape(), kFloat32, kCPU);
    }
}

TensorPtr PositionalEmbeddingLayer::forward(const TensorPtr& input, bool training) {
    if (!input) throw std::runtime_error("PositionalEmbeddingLayer::forward: input is null");
    int64_t seq_len = input->shape()[0];
    return forward(static_cast<int>(seq_len), training);
}

TensorPtr PositionalEmbeddingLayer::forward(int sequence_length, bool training) {
    if (sequence_length > max_position_) {
        throw std::runtime_error("PositionalEmbeddingLayer: sequence length exceeds maximum position");
    }
    auto out = zeros({sequence_length, embedding_dim_}, kFloat32, kCPU);
    const float* src = position_embeddings_->data<float>();
    float* dst = out->data<float>();
    std::memcpy(dst, src, static_cast<size_t>(sequence_length * embedding_dim_) * sizeof(float));
    if (training && trainable_) {
        last_seq_len_ = sequence_length;
        last_output_ = out;
    }
    return out;
}

std::vector<TensorPtr> PositionalEmbeddingLayer::get_parameters() const {
    return {position_embeddings_};
}

std::vector<TensorPtr> PositionalEmbeddingLayer::get_gradients() const {
    if (trainable_ && grad_position_embeddings_) return {grad_position_embeddings_};
    return {};
}

void PositionalEmbeddingLayer::zero_grad() {
    if (grad_position_embeddings_) {
        std::memset(grad_position_embeddings_->data_ptr(), 0,
                    static_cast<size_t>(grad_position_embeddings_->numel() * sizeof(float)));
    }
}

void PositionalEmbeddingLayer::update_parameters(const std::vector<TensorPtr>& updates) {
    if (!trainable_ || updates.empty() || !updates[0]) return;
    if (updates[0]->shape() != position_embeddings_->shape()) {
        throw std::runtime_error("PositionalEmbeddingLayer::update_parameters: shape mismatch");
    }
    float* w = position_embeddings_->data<float>();
    const float* u = updates[0]->data<float>();
    for (int64_t i = 0; i < position_embeddings_->numel(); ++i) w[i] += u[i];
}

void PositionalEmbeddingLayer::save_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    f.write(reinterpret_cast<const char*>(&max_position_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(int));
    const float* buf = position_embeddings_->data<float>();
    f.write(reinterpret_cast<const char*>(buf),
            static_cast<std::streamsize>(position_embeddings_->numel() * sizeof(float)));
}

void PositionalEmbeddingLayer::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);
    int mp, ed;
    f.read(reinterpret_cast<char*>(&mp), sizeof(int));
    f.read(reinterpret_cast<char*>(&ed), sizeof(int));
    if (mp != max_position_ || ed != embedding_dim_) {
        throw std::runtime_error("PositionalEmbeddingLayer::load_weights: config mismatch");
    }
    float* buf = position_embeddings_->data<float>();
    f.read(reinterpret_cast<char*>(buf),
           static_cast<std::streamsize>(position_embeddings_->numel() * sizeof(float)));
}

size_t PositionalEmbeddingLayer::get_param_count() const {
    return static_cast<size_t>(position_embeddings_->numel());
}

}
