#pragma once

#include "../core/lm_loss.h"
#include "../core/memory_manager.h"
#include "../graph/gemma_model.h"
#include "../graph/gpt2_model.h"
#include "../graph/qwen_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace ops::smoke {

inline void fill_constant(const TensorPtr& tensor, float value) {
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = value;
    }
}

inline void fill_uniform(const TensorPtr& tensor, float low, float high, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(low, high);
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = dist(rng);
    }
}

inline void fill_ids(const TensorPtr& tensor, int vocab_size, int start_token = 1) {
    int32_t* data = tensor->data<int32_t>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = static_cast<int32_t>((start_token + i) % std::max(vocab_size, 2));
    }
}

inline TensorPtr make_input_ids(int batch_size, int seq_len, int vocab_size, int start_token = 1) {
    auto ids = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len}, kInt32, kCPU);
    fill_ids(ids, vocab_size, start_token);
    return ids;
}

inline TensorPtr make_shifted_labels(const TensorPtr& input_ids, int vocab_size) {
    auto labels = std::make_shared<Tensor>(input_ids->shape(), kInt32, kCPU);
    const int32_t* src = input_ids->data<int32_t>();
    int32_t* dst = labels->data<int32_t>();
    for (int64_t i = 0; i < input_ids->numel(); ++i) {
        dst[i] = static_cast<int32_t>((src[i] + 1) % std::max(vocab_size, 2));
    }
    return labels;
}

inline TensorPtr make_attention_mask(int batch_size, int seq_len) {
    auto mask = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len}, kFloat32, kCPU);
    fill_constant(mask, 1.0f);
    return mask;
}

inline bool is_finite_scalar(const TensorPtr& tensor) {
    return tensor && tensor->numel() == 1 &&
           std::isfinite(static_cast<double>(tensor->data<float>()[0]));
}

inline double grad_l2_norm(const std::vector<TensorPtr>& grads) {
    double accum = 0.0;
    for (const auto& grad : grads) {
        if (!grad) {
            continue;
        }
        const float* data = grad->data<float>();
        for (int64_t i = 0; i < grad->numel(); ++i) {
            accum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
        }
    }
    return std::sqrt(accum);
}

inline double max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* da = a->data<float>();
    const float* db = b->data<float>();
    double max_diff = 0.0;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(da[i]) - static_cast<double>(db[i])));
    }
    return max_diff;
}

inline double max_param_delta(const std::vector<TensorPtr>& before, const std::vector<TensorPtr>& after) {
    double max_delta = 0.0;
    for (size_t i = 0; i < before.size() && i < after.size(); ++i) {
        max_delta = std::max(max_delta, max_abs_diff(before[i], after[i]));
    }
    return max_delta;
}

inline std::vector<TensorPtr> clone_tensors(const std::vector<TensorPtr>& tensors) {
    std::vector<TensorPtr> clones;
    clones.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        clones.push_back(std::make_shared<Tensor>(*tensor));
    }
    return clones;
}

inline void initialize_tiny_gpt2(GPT2Model& model, std::mt19937& rng) {
    auto params = model.parameters();
    size_t idx = 0;

    fill_uniform(params[idx++], -0.05f, 0.05f, rng);
    fill_uniform(params[idx++], -0.05f, 0.05f, rng);

    const int layers = model.config().n_layer;
    for (int layer = 0; layer < layers; ++layer) {
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
    }

    fill_constant(params[idx++], 1.0f);
    fill_constant(params[idx++], 0.0f);
}

inline void initialize_tiny_qwen(QwenModel& model, std::mt19937& rng) {
    auto params = model.parameters();
    size_t idx = 0;

    fill_uniform(params[idx++], -0.05f, 0.05f, rng);
    fill_constant(params[idx++], 1.0f);

    const int layers = model.config().num_hidden_layers;
    for (int layer = 0; layer < layers; ++layer) {
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 1.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 0.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
    }
}

inline void initialize_tiny_gemma(GemmaModel& model, std::mt19937& rng) {
    auto params = model.parameters();
    size_t idx = 0;

    fill_uniform(params[idx++], -0.05f, 0.05f, rng);
    fill_constant(params[idx++], 1.0f);
    fill_uniform(params[idx++], -0.05f, 0.05f, rng);

    const int layers = model.config().num_hidden_layers;
    for (int layer = 0; layer < layers; ++layer) {
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 1.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_constant(params[idx++], 1.0f);
        fill_constant(params[idx++], 1.0f);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
        fill_uniform(params[idx++], -0.05f, 0.05f, rng);
    }
}

inline void zero_grads(const std::vector<TensorPtr>& params) {
    for (const auto& param : params) {
        if (param) {
            param->zero_grad();
        }
    }
}

inline void cleanup_step_memory() {
    MemoryManager::instance().force_cleanup();
}

}  // namespace ops::smoke
