#pragma once

#include "../core/tensor.h"
#include "lora_ops.h"
#include <memory>

namespace ops {

struct LoRAConfig : public LayerConfig {
    int rank;
    float alpha;
    float dropout;
    bool use_bias;
    bool residual;

    LoRAConfig() : LayerConfig(),
        rank(8), alpha(32.0f), dropout(0.1f), use_bias(false), residual(true) {}

    LoRAConfig(
        int in_features,
        int out_features,
        int rank_ = 8,
        float alpha_ = 32.0f,
        float dropout_ = 0.1f,
        bool use_bias_ = false,
        bool residual_ = true
    ) : LayerConfig(in_features, out_features, use_bias_, dropout_),
        rank(rank_), alpha(alpha_), dropout(dropout_), use_bias(use_bias_), residual(residual_) {}

    LoRAOpsConfig to_ops_config() const {
        return LoRAOpsConfig(rank, alpha, dropout, use_bias, residual);
    }
};

class LoRALayer : public BaseLayer {
public:
    explicit LoRALayer(const LoRAConfig& config);
    ~LoRALayer() override = default;

    TensorPtr forward(const Tensor& input, bool training = false) override;

    TensorPtr backward(const Tensor& grad_output) override;

    std::vector<TensorPtr> get_parameters() override;
    std::vector<TensorPtr> get_gradients() override;

    void update_parameters(const std::vector<TensorPtr>& updates) override;

    void save_weights(const std::string& path) const override;
    void load_weights(const std::string& path) override;

    size_t get_param_count() const override;

    const LoRAConfig& get_lora_config() const { return lora_config_; }

    void merge_with_base_weights(const Tensor& base_weights);
    void save_lora_weights(const std::string& path) const;
    void load_lora_weights(const std::string& path);

protected:
    void init_parameters() override;

    virtual TensorPtr forward_lora(const Tensor& input, bool training);

    virtual TensorPtr backward_lora(const Tensor& grad_output);

protected:
    LoRAConfig lora_config_;

    TensorPtr lora_A_;
    TensorPtr lora_B_;
    TensorPtr bias_;

    TensorPtr grad_lora_A_;
    TensorPtr grad_lora_B_;
    TensorPtr grad_bias_;

    TensorPtr last_input_;
    TensorPtr last_hidden_;

    TensorPtr apply_dropout(const Tensor& x, bool training) override;
    float get_scale_factor() const {
        return (lora_config_.rank > 0) ? (lora_config_.alpha / lora_config_.rank) : 0.0f;
    }
};

}
