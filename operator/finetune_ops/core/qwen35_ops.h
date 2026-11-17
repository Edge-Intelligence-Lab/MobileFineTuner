#pragma once
#include "tensor.h"

namespace ops {
// CPU FP32 reference kernels with complete reverse-mode derivatives.
TensorPtr causal_depthwise_conv1d(const TensorPtr& input, const TensorPtr& weight);
TensorPtr partial_rotary_embedding(const TensorPtr& input, int rotary_dim, float theta);
TensorPtr gated_delta_rule(const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
                           const TensorPtr& decay_projection, const TensorPtr& beta_projection,
                           const TensorPtr& log_decay, const TensorPtr& dt_bias);
} // namespace ops
