/**
 * @file lora_injector.h
 * @brief LoRA injector (supports split_qkv, merge/unmerge, save/load)
 */

#pragma once

#include "../core/tensor.h"
#include "gpt2_model.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace ops {

/**
 * @brief LoRA target layer types
 */
enum class LoraTarget {
    AttnQKV,    // Attention QKV (can be split into q/k/v)
    AttnProj,   // Attention output projection
    MlpFcIn,    // MLP fc_in (C → 4C)
    MlpFcOut    // MLP fc_out (4C → C)
};

/**
 * @brief LoRA configuration
 */
struct LoraSpec {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.05f;
    bool split_qkv = true;  // Whether to split QKV into q/k/v (recommended)
    
    // By default apply LoRA only to attention (matches PyTorch PEFT convention)
    // To include MLP, manually add LoraTarget::MlpFcIn and LoraTarget::MlpFcOut
    std::vector<LoraTarget> targets = {
        LoraTarget::AttnQKV,   // Attention Q/K/V projection
        LoraTarget::AttnProj   // Attention output projection
    };
    
    std::vector<int> layers;  // empty = all layers
    
    LoraSpec() = default;
    
    // Default config (attention-only)
    static LoraSpec default_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        return spec;
    }
    
    // Config including MLP (broader coverage)
    static LoraSpec full_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        spec.targets = {
            LoraTarget::AttnQKV,
            LoraTarget::AttnProj,
            LoraTarget::MlpFcIn,
            LoraTarget::MlpFcOut
        };
        return spec;
    }
};

/**
 * @brief Single LoRA state (A/B matrices + metadata)
 */
struct LoraState {
    TensorPtr A;  // [in, r]
    TensorPtr B;  // [r, out]
    float scale;  // alpha / r
    float dropout_p;
    bool enabled = true;
    
    // Initialize (A ~ N(0, 1/r), B = 0)
    void init(int64_t in_features, int64_t out_features, int rank, float alpha, float dropout);
};

/**
 * @brief LoRA injector
 */
class LoraInjector {
public:
    LoraInjector() = default;
    ~LoraInjector() = default;
    
    /**
     * @brief Inject LoRA into the model and freeze base weights
     * @param model GPT2Model instance
     * @param spec LoRA configuration
     */
    void inject(GPT2Model& model, const LoraSpec& spec);
    
    /**
     * @brief Merge LoRA into base weights (pre-inference)
     * W' = W + B @ A * scale
     */
    void merge();
    
    /**
     * @brief Unmerge LoRA (resume training)
     * W = W' - B @ A * scale
     */
    void unmerge();
    
    /**
     * @brief Merge all LoRALinear LoRA weights into base
     */
    void merge_all(GPT2Model& model);
    
    /**
     * @brief Unmerge all LoRALinear LoRA weights
     */
    void unmerge_all(GPT2Model& model);
    
    /**
     * @brief Collect trainable parameters (LoRA A/B only)
     * @return LoRA parameter list
     */
    std::vector<TensorPtr> collect_lora_parameters() const;
    
    /**
     * @brief Save LoRA weights to safetensors
     * @param path output path
     */
    void save_lora_safetensors(const std::string& path) const;
    
    /**
     * @brief Load LoRA weights from safetensors
     * @param path input path
     */
    void load_lora_safetensors(const std::string& path);
    
    /**
     * @brief Print LoRA injection info
     */
    void print_info() const;
    
    /**
     * @brief Get all LoRA trainable parameters (A and B matrices)
     * @return List of all LoRA parameters
     */
    std::vector<TensorPtr> get_trainable_params();
    
    /**
     * @brief LoRA-augmented linear forward (wrapper)
     * @param x input [*, in]
     * @param W base weight [in, out]
     * @param bias base bias [out] (optional)
     * @param lora LoRA state (optional)
     * @param training training mode flag (affects dropout)
     * @return output [*, out]
     */
    static TensorPtr lora_linear_forward(const TensorPtr& x,
                                        const TensorPtr& W,
                                        const TensorPtr& bias,
                                        const LoraState* lora,
                                        bool training = false);

private:
    struct Hook {
        std::string name;
        TensorPtr* W_ptr;
        TensorPtr* bias_ptr;
        LoraState state;
        // Column range when applying to partial weight (e.g., split Q/K/V)
        // Range: [col_offset, col_offset + col_size)
        int64_t col_offset = 0;
        int64_t col_size = -1;  // -1 means cover entire out dimension
    };
    
    std::vector<Hook> hooks_;
    LoraSpec spec_;
    bool merged_ = false;
    int num_layers_ = 0;
    
    // Internal helpers
    void inject_qkv_split(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_qkv_fused(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_layer(GPT2Model& model, int layer_idx, const std::string& layer_name,
                     int64_t in_features, int64_t out_features,
                     int rank, float alpha, float dropout);
};

}  // namespace ops

