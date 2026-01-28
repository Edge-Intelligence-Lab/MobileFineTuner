#pragma once

#include "tensor.h"
#include "basic_ops.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace ops {

struct SamplingConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 0.0f;
    float repetition_penalty = 1.0f;
    std::vector<int> used_tokens;

    SamplingConfig() = default;
    SamplingConfig(float temp, int k, float p, float penalty)
        : temperature(temp), top_k(k), top_p(p), repetition_penalty(penalty) {}
};

class AutoregressiveOps {
public:

    static int generate_next_token(const Tensor& logits, const SamplingConfig& config);

    static std::vector<float> get_next_token_probs(const Tensor& logits, const SamplingConfig& config);

    static std::vector<float> apply_temperature(const std::vector<float>& logits, float temperature);

    static std::vector<float> top_k_sampling(const std::vector<float>& logits, int k);

    static std::vector<float> top_p_sampling(const std::vector<float>& logits, float p);

    static std::vector<float> apply_repetition_penalty(const std::vector<float>& logits,
                                                      const std::vector<int>& used_tokens,
                                                      float penalty);

    static int sample_from_probs(const std::vector<float>& probs);

    static std::vector<float> logits_to_probs(const std::vector<float>& logits);

private:
    static std::mt19937& get_random_generator();
};

inline int simple_generate_next_token(const Tensor& logits, float temperature = 1.0f) {
    std::vector<float> logits_vec(logits.data.begin(), logits.data.end());

    if (temperature != 1.0f) {
        for (float& val : logits_vec) {
            val /= temperature;
        }
    }

    auto probs = AutoregressiveOps::logits_to_probs(logits_vec);

    return AutoregressiveOps::sample_from_probs(probs);
}

}