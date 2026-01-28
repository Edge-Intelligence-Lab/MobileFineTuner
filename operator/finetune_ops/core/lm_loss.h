/**
 * @file lm_loss.h
 * @brief Language-model-specific loss functions (supports 3D logits and ignore_index)
 */

#pragma once

#include "tensor.h"
#include <string>
#include <cmath>

namespace ops {

/**
 * @brief Cross-Entropy loss for language modeling
 * 
 * @param logits [B, S, V] float32
 * @param labels [B, S] int32, PAD positions use ignore_index
 * @param ignore_index Label value to ignore (default -100)
 * @param reduction "mean" | "sum" | "none"
 * @return 
 *   - "mean": scalar loss averaged over valid tokens
 *   - "sum": scalar loss summed over valid tokens
 *   - "none": [B,S] per-token NLL
 * 
 * Features:
 * - Numerically stable (logsumexp)
 * - Automatically skips ignore_index (excluded from loss/grad)
 * - Autograd supported
 */
TensorPtr lm_cross_entropy(const TensorPtr& logits,
                          const TensorPtr& labels,
                          int ignore_index = -100,
                          const std::string& reduction = "mean");

/**
 * @brief Compute perplexity from mean NLL
 */
inline float perplexity_from_loss(float mean_nll) {
    return std::exp(mean_nll);
}

}  // namespace ops

