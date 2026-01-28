#pragma once

#include "optimizer.h"
#include "../core/tensor.h"
#include "../layer/lora_layer.h"
#include "../layer/lora_attention_layer.h"
#include "../layer/lora_lm_head_layer.h"
#include <vector>
#include <memory>

namespace ops {

struct LoRAOptimizerConfig : public OptimizerConfig {
    float weight_decay;
    float max_grad_norm;
    bool only_lora_params;

    LoRAOptimizerConfig()
        : OptimizerConfig(),
          weight_decay(0.01f),
          max_grad_norm(1.0f),
          only_lora_params(true) {}

    LoRAOptimizerConfig(
        float learning_rate,
        float weight_decay = 0.01f,
        float max_grad_norm = 1.0f,
        bool only_lora_params = true
    ) : OptimizerConfig(learning_rate),
        weight_decay(weight_decay),
        max_grad_norm(max_grad_norm),
        only_lora_params(only_lora_params) {}
};

class LoRAOptimizer {
public:
    explicit LoRAOptimizer(const LoRAOptimizerConfig& config);

    void add_lora_layer(LoRALayer* layer);
    void add_attention_layer(LoRAAttentionLayer* layer);
    void add_lm_head_layer(LoRALMHeadLayer* layer);

    void step();

    void zero_grad();

    void clip_grad_norm();

    float get_learning_rate() const;
    void set_learning_rate(float lr);

private:
    LoRAOptimizerConfig config_;
    std::unique_ptr<Optimizer> optimizer_;

    std::vector<LoRALayer*> lora_layers_;
    std::vector<LoRAAttentionLayer*> attention_layers_;
    std::vector<LoRALMHeadLayer*> lm_head_layers_;

    std::vector<TensorPtr> collect_parameters();
    std::vector<TensorPtr> collect_gradients();
    float compute_grad_norm();
    void scale_gradients(float scale);
};

}
