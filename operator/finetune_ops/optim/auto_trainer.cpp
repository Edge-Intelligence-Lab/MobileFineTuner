#include "auto_trainer.h"

#include "../core/lm_loss.h"

#include <stdexcept>
#include <vector>

namespace ops {

AutoTrainer::AutoTrainer(AutoModelForCausalLM& model,
                         const AutoTrainerConfig& config)
    : model_(model), config_(config) {
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

AutoTrainStepResult AutoTrainer::train_step(const TensorPtr& input_ids,
                                            const TensorPtr& attention_mask,
                                            const TensorPtr& labels) {
    auto params = model_.trainable_parameters();
    if (params.empty()) {
        throw std::runtime_error("AutoTrainer::train_step requires trainable parameters; call init_lora first");
    }

    auto logits = model_.forward(input_ids, attention_mask);
    auto loss = lm_cross_entropy(logits, labels, config_.ignore_index, "mean");
    loss->backward();

    optimizer_->clip_grad_norm(params, config_.max_grad_norm);

    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(params, grads);
    optimizer_->zero_grad(params);

    AutoTrainStepResult result;
    result.loss = loss->data<float>()[0];
    result.trainable_tensor_count = static_cast<int>(params.size());
    return result;
}

}  // namespace ops
