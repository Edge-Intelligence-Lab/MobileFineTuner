#include "lora_optimizer.h"
#include "adam.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include <cmath>

namespace ops {

LoRAOptimizer::LoRAOptimizer(const LoRAOptimizerConfig& config)
    : config_(config) {

    AdamConfig adam_config(
        config.learning_rate,
        0.9f,
        0.999f,
        1e-8f,
        config.weight_decay,
        config.max_grad_norm
    );
    optimizer_ = std::make_unique<Adam>(adam_config);
}

void LoRAOptimizer::add_lora_layer(LoRALayer* layer) {
    if (layer) {
        lora_layers_.push_back(layer);
    }
}

void LoRAOptimizer::add_attention_layer(LoRAAttentionLayer* layer) {
    if (layer) {
        attention_layers_.push_back(layer);
    }
}

void LoRAOptimizer::add_lm_head_layer(LoRALMHeadLayer* layer) {
    if (layer) {
        lm_head_layers_.push_back(layer);
    }
}

void LoRAOptimizer::step() {

    auto parameters = collect_parameters();
    auto gradients = collect_gradients();

    if (config_.max_grad_norm > 0.0f) {
        clip_grad_norm();
    }

    optimizer_->step(parameters, gradients);
}

void LoRAOptimizer::zero_grad() {

    for (auto layer : lora_layers_) {
        for (auto& grad : layer->get_gradients()) {
            grad->zero_();
        }
    }

    for (auto layer : attention_layers_) {
        for (auto& grad : layer->get_gradients()) {
            grad->zero_();
        }
    }

    for (auto layer : lm_head_layers_) {
        for (auto& grad : layer->get_gradients()) {
            grad->zero_();
        }
    }
}

void LoRAOptimizer::clip_grad_norm() {
    float total_norm = compute_grad_norm();

    if (total_norm > config_.max_grad_norm) {
        float scale = config_.max_grad_norm / (total_norm + 1e-6f);
        scale_gradients(scale);
    }
}

float LoRAOptimizer::get_learning_rate() const {
    return config_.learning_rate;
}

void LoRAOptimizer::set_learning_rate(float lr) {
    config_.learning_rate = lr;
    optimizer_->set_learning_rate(lr);
}

std::vector<TensorPtr> LoRAOptimizer::collect_parameters() {
    std::vector<TensorPtr> parameters;

    for (auto layer : lora_layers_) {
        auto layer_params = layer->get_parameters();
        parameters.insert(parameters.end(),
                        layer_params.begin(),
                        layer_params.end());
    }

    for (auto layer : attention_layers_) {
        if (!config_.only_lora_params) {
            auto layer_params = layer->get_parameters();
            parameters.insert(parameters.end(),
                            layer_params.begin(),
                            layer_params.end());
        } else {

            if (layer->get_query_lora()) {
                auto lora_params = layer->get_query_lora()->get_parameters();
                parameters.insert(parameters.end(),
                                lora_params.begin(),
                                lora_params.end());
            }
            if (layer->get_key_lora()) {
                auto lora_params = layer->get_key_lora()->get_parameters();
                parameters.insert(parameters.end(),
                                lora_params.begin(),
                                lora_params.end());
            }
            if (layer->get_value_lora()) {
                auto lora_params = layer->get_value_lora()->get_parameters();
                parameters.insert(parameters.end(),
                                lora_params.begin(),
                                lora_params.end());
            }
            if (layer->get_output_lora()) {
                auto lora_params = layer->get_output_lora()->get_parameters();
                parameters.insert(parameters.end(),
                                lora_params.begin(),
                                lora_params.end());
            }
        }
    }

    for (auto layer : lm_head_layers_) {
        if (!config_.only_lora_params) {
            auto layer_params = layer->get_parameters();
            parameters.insert(parameters.end(),
                            layer_params.begin(),
                            layer_params.end());
        } else {
            if (layer->get_lora()) {
                auto lora_params = layer->get_lora()->get_parameters();
                parameters.insert(parameters.end(),
                                lora_params.begin(),
                                lora_params.end());
            }
        }
    }

    return parameters;
}

std::vector<TensorPtr> LoRAOptimizer::collect_gradients() {
    std::vector<TensorPtr> gradients;

    for (auto layer : lora_layers_) {
        auto layer_grads = layer->get_gradients();
        gradients.insert(gradients.end(),
                        layer_grads.begin(),
                        layer_grads.end());
    }

    for (auto layer : attention_layers_) {
        if (!config_.only_lora_params) {
            auto layer_grads = layer->get_gradients();
            gradients.insert(gradients.end(),
                            layer_grads.begin(),
                            layer_grads.end());
        } else {

            if (layer->get_query_lora()) {
                auto lora_grads = layer->get_query_lora()->get_gradients();
                gradients.insert(gradients.end(),
                               lora_grads.begin(),
                               lora_grads.end());
            }
            if (layer->get_key_lora()) {
                auto lora_grads = layer->get_key_lora()->get_gradients();
                gradients.insert(gradients.end(),
                               lora_grads.begin(),
                               lora_grads.end());
            }
            if (layer->get_value_lora()) {
                auto lora_grads = layer->get_value_lora()->get_gradients();
                gradients.insert(gradients.end(),
                               lora_grads.begin(),
                               lora_grads.end());
            }
            if (layer->get_output_lora()) {
                auto lora_grads = layer->get_output_lora()->get_gradients();
                gradients.insert(gradients.end(),
                               lora_grads.begin(),
                               lora_grads.end());
            }
        }
    }

    for (auto layer : lm_head_layers_) {
        if (!config_.only_lora_params) {
            auto layer_grads = layer->get_gradients();
            gradients.insert(gradients.end(),
                            layer_grads.begin(),
                            layer_grads.end());
        } else {
            if (layer->get_lora()) {
                auto lora_grads = layer->get_lora()->get_gradients();
                gradients.insert(gradients.end(),
                               lora_grads.begin(),
                               lora_grads.end());
            }
        }
    }

    return gradients;
}

float LoRAOptimizer::compute_grad_norm() {
    float total_norm = 0.0f;
    auto gradients = collect_gradients();

    for (const auto& grad : gradients) {
        float norm = 0.0f;
        for (const auto& val : grad->data()) {
            norm += val * val;
        }
        total_norm += norm;
    }

    return std::sqrt(total_norm);
}

void LoRAOptimizer::scale_gradients(float scale) {
    auto gradients = collect_gradients();

    for (auto& grad : gradients) {
        for (auto& val : grad->data()) {
            val *= scale;
        }
    }
}

}
