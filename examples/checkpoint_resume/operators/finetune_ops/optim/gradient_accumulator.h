#pragma once

#include "../core/tensor.h"
#include "../data/wikitext2_dataset.h"
#include <stdexcept>
#include <vector>

namespace ops {

inline Batch own_microbatch(const Batch& batch) {
    // Dataset batches are views into reusable buffers, invalidated by next_batch().
    return {batch.input_ids->clone(), batch.attention_mask->clone(), batch.labels->clone()};
}

inline int64_t valid_lm_label_count(const TensorPtr& labels, int64_t vocab_size) {
    if (!labels || labels->ndim() != 2 || labels->dtype() != kInt32)
        throw std::runtime_error("Expected int32 [B,S] LM labels");
    int64_t count = 0;
    const int64_t sequence = labels->shape()[1];
    for (int64_t b = 0; b < labels->shape()[0]; ++b) {
        for (int64_t s = 1; s < sequence; ++s) {
            const auto label = labels->data<int32_t>()[b * sequence + s];
            count += label >= 0 && label < vocab_size;
        }
    }
    return count;
}

// Engine::run_backward writes (rather than adds) leaf grads on each call.
// Preserve each microbatch's contribution outside the autograd graph.
class GradientAccumulator {
public:
    explicit GradientAccumulator(size_t count) : sums_(count) {}
    void collect(const std::vector<TensorPtr>& parameters) {
        if (parameters.size() != sums_.size()) throw std::runtime_error("Gradient accumulator size mismatch");
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto grad = parameters[i]->grad();
            if (!grad) throw std::runtime_error("Missing microbatch LoRA gradient");
            if (!sums_[i]) {
                sums_[i] = grad->clone();
            } else {
                for (int64_t j = 0; j < grad->numel(); ++j)
                    sums_[i]->data<float>()[j] += grad->data<float>()[j];
            }
            parameters[i]->zero_grad();
        }
    }
    void assign(const std::vector<TensorPtr>& parameters) const {
        if (parameters.size() != sums_.size()) throw std::runtime_error("Gradient accumulator size mismatch");
        for (size_t i = 0; i < parameters.size(); ++i) parameters[i]->set_grad(sums_[i]);
    }
private:
    std::vector<TensorPtr> sums_;
};

} // namespace ops
