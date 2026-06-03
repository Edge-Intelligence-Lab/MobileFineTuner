#pragma once

#include "../core/tensor.h"
#include "../graph/auto_model.h"
#include "adam.h"

#include <memory>

namespace ops {

struct AutoTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    int ignore_index = -100;
};

struct AutoTrainStepResult {
    float loss = 0.0f;
    int trainable_tensor_count = 0;
};

class AutoTrainer {
public:
    AutoTrainer(AutoModelForCausalLM& model,
                const AutoTrainerConfig& config = AutoTrainerConfig());

    AutoTrainStepResult train_step(const TensorPtr& input_ids,
                                   const TensorPtr& attention_mask,
                                   const TensorPtr& labels);

private:
    AutoModelForCausalLM& model_;
    AutoTrainerConfig config_;
    std::unique_ptr<Adam> optimizer_;
};

}  // namespace ops
