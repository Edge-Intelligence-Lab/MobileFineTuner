#include "auto_trainer.h"

#include "../core/autograd_engine.h"
#include "../core/lm_loss.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ops {

AutoTrainer::AutoTrainer(AutoModelForCausalLM& model,
                         const AutoTrainerConfig& config)
    : model_(model), config_(config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate < 0 ||
        !std::isfinite(config.weight_decay) || config.weight_decay < 0 ||
        !std::isfinite(config.max_grad_norm) || config.max_grad_norm < 0)
        throw std::invalid_argument("AutoTrainer requires finite nonnegative learning_rate, "
                                    "weight_decay and max_grad_norm");
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

    try {
        auto logits = model_.forward(input_ids, attention_mask);
        auto loss = lm_cross_entropy(logits, labels, config_.ignore_index, "mean");
        if (!std::isfinite(loss->data<float>()[0]))
            throw std::runtime_error("Non-finite training loss");
        loss->backward();

        if (config_.max_grad_norm > 0)
            optimizer_->clip_grad_norm(params, config_.max_grad_norm);

        std::vector<TensorPtr> grads;
        grads.reserve(params.size());
        for (const auto& param : params) {
            const auto gradient = param->grad();
            if (!gradient || gradient->shape() != param->shape())
                throw std::runtime_error("Missing or incompatible trainable gradient");
            for (int64_t i = 0; i < gradient->numel(); ++i)
                if (!std::isfinite(gradient->data<float>()[i]))
                    throw std::runtime_error("Non-finite trainable gradient");
            grads.push_back(gradient);
        }
        optimizer_->step(params, grads);
        optimizer_->zero_grad(params);

        AutoTrainStepResult result;
        result.loss = loss->data<float>()[0];
        result.trainable_tensor_count = static_cast<int>(params.size());
        return result;
    } catch (...) {
        autograd::Engine::instance().clear_graph();
        optimizer_->zero_grad(params);
        throw;
    }
}

}  // namespace ops
