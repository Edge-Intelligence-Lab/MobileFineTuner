/**
 * @file lm_loss.h
 * @brief Language model specialized loss functions (supports 3D logits and ignore_index)
 */

#pragma once

#include "tensor.h"
#include <string>
#include <cmath>

namespace ops {

/**
 * @brief Language Model Cross-Entropy Loss
 * 
 * @param logits [B, S, V] float32
 * @param labels [B, S] int32, PAD positions are ignore_index
 * @param ignore_index Label value to ignore (default -100)
 * @param reduction "mean" | "sum" | "none"
 * @return 
 *   - "mean": Scalar loss (averaged over valid tokens)
 *   - "sum": Scalar loss (summed over valid tokens)
 *   - "none": [B,S] per-token NLL
 * 
 * Features:
 * - Numerically stable (logsumexp)
 * - Automatically ignores ignore_index (excluded from loss and gradient)
 * - Supports automatic differentiation
 */
TensorPtr lm_cross_entropy(const TensorPtr& logits,
                          const TensorPtr& labels,
                          int ignore_index = -100,
                          const std::string& reduction = "mean");

/**
 * @brief Calculate perplexity from mean NLL
 */
inline float perplexity_from_loss(float mean_nll) {
    return std::exp(mean_nll);
}

}  // namespace ops

