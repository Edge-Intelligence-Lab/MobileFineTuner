#include "autoregressive_ops.h"
#include <iostream>
#include <numeric>
#include <unordered_set>

namespace ops {

std::mt19937& AutoregressiveOps::get_random_generator() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

int AutoregressiveOps::generate_next_token(const Tensor& logits, const SamplingConfig& config) {
    auto probs = get_next_token_probs(logits, config);
    return sample_from_probs(probs);
}

std::vector<float> AutoregressiveOps::get_next_token_probs(const Tensor& logits, const SamplingConfig& config) {
    std::vector<float> logits_vec(logits.data.begin(), logits.data.end());

    if (config.repetition_penalty != 1.0f && !config.used_tokens.empty()) {
        logits_vec = apply_repetition_penalty(logits_vec, config.used_tokens, config.repetition_penalty);
    }

    logits_vec = apply_temperature(logits_vec, config.temperature);

    if (config.top_k > 0) {
        logits_vec = top_k_sampling(logits_vec, config.top_k);
    }

    if (config.top_p > 0.0f) {
        logits_vec = top_p_sampling(logits_vec, config.top_p);
    }

    return logits_to_probs(logits_vec);
}

std::vector<float> AutoregressiveOps::apply_temperature(const std::vector<float>& logits, float temperature) {
    if (temperature == 1.0f) {
        return logits;
    }

    std::vector<float> result = logits;
    for (float& val : result) {
        val /= temperature;
    }
    return result;
}

std::vector<float> AutoregressiveOps::top_k_sampling(const std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) {
        return logits;
    }

    std::vector<std::pair<float, int>> indexed_logits;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        indexed_logits.emplace_back(logits[i], i);
    }

    std::partial_sort(indexed_logits.begin(), indexed_logits.begin() + k, indexed_logits.end(),
                     std::greater<std::pair<float, int>>());

    std::vector<float> result(logits.size(), -1e9f);
    for (int i = 0; i < k; ++i) {
        result[indexed_logits[i].second] = indexed_logits[i].first;
    }

    return result;
}

std::vector<float> AutoregressiveOps::top_p_sampling(const std::vector<float>& logits, float p) {
    if (p <= 0.0f || p >= 1.0f) {
        return logits;
    }

    std::vector<std::pair<float, int>> indexed_logits;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        indexed_logits.emplace_back(logits[i], i);
    }

    std::sort(indexed_logits.begin(), indexed_logits.end(), std::greater<std::pair<float, int>>());

    auto probs = logits_to_probs(logits);

    float cumulative_prob = 0.0f;
    std::vector<float> result(logits.size(), -1e9f);

    for (const auto& pair : indexed_logits) {
        int idx = pair.second;
        cumulative_prob += probs[idx];
        result[idx] = pair.first;

        if (cumulative_prob >= p) {
            break;
        }
    }

    return result;
}

std::vector<float> AutoregressiveOps::apply_repetition_penalty(const std::vector<float>& logits,
                                                              const std::vector<int>& used_tokens,
                                                              float penalty) {
    if (penalty == 1.0f || used_tokens.empty()) {
        return logits;
    }

    std::vector<float> result = logits;
    std::unordered_set<int> used_set(used_tokens.begin(), used_tokens.end());

    for (int token_id : used_set) {
        if (token_id >= 0 && token_id < static_cast<int>(result.size())) {
            result[token_id] *= penalty;
        }
    }

    return result;
}

int AutoregressiveOps::sample_from_probs(const std::vector<float>& probs) {

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) {

        std::unifor_int_distribution<int> dist(0, static_cast<int>(probs.size()) - 1);
        return dist(get_random_generator());
    }

    std::vector<float> normalized_probs = probs;
    for (float& prob : normalized_probs) {
        prob /= sum;
    }

    std::vector<float> cumulative_probs(normalized_probs.size());
    std::partial_sum(normalized_probs.begin(), normalized_probs.end(), cumulative_probs.begin());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_val = dist(get_random_generator());

    for (int i = 0; i < static_cast<int>(cumulative_probs.size()); ++i) {
        if (random_val <= cumulative_probs[i]) {
            return i;
        }
    }

    return static_cast<int>(probs.size()) - 1;
}

std::vector<float> AutoregressiveOps::logits_to_probs(const std::vector<float>& logits) {

    float max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<float> exp_logits(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        exp_logits[i] = std::exp(logits[i] - max_logit);
    }

    float sum = std::accumulate(exp_logits.begin(), exp_logits.end(), 0.0f);

    std::vector<float> probs(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        probs[i] = exp_logits[i] / sum;
    }

    return probs;
}

}