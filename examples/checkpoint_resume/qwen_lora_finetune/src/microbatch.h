#pragma once

#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/optim/gradient_accumulator.h"
#include <algorithm>

namespace ops::microbatch {

template<class Consumer>
void for_each_chunk(const Batch& batch, int physical, Consumer consume) {
    const auto rows = batch.input_ids->size(0);
    for (int64_t first = 0; first < rows; first += physical) {
        const auto end = std::min<int64_t>(first + physical, rows);
        consume(Batch{batch.input_ids->slice(0, first, end),
                      batch.attention_mask->slice(0, first, end),
                      batch.labels->slice(0, first, end)});
    }
}

// Evaluation averages logical batch means, not physical chunk means.
template<class Model>
double logical_mean_loss(Model& model, const Batch& batch, int physical) {
    double sum = 0;
    int64_t total = 0;
    for_each_chunk(batch, physical, [&](const Batch& chunk) {
        const auto n = valid_lm_label_count(chunk.labels, model.config().vocab_size);
        if (!n) return;
        auto logits = model.forward(chunk.input_ids, chunk.attention_mask);
        TensorPtr loss = lm_cross_entropy(logits, chunk.labels, -100, "mean");
        sum += static_cast<double>(loss->data<float>()[0]) * n;
        total += n;
    });
    return total ? sum / total : 0.0;
}

}  // namespace ops::microbatch
