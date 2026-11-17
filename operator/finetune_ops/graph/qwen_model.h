/**
 * @file qwen_model.h
 * @brief Configuration-driven Qwen2/Qwen3 dense and Qwen3.5 hybrid text
 * decoder.
 *
 * CPU/BLAS text LoRA training with architecture-specific mixer dispatch.
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"
#include "frozen_model_offload.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops {

struct SafeTensorsLoadOptions;

struct QwenConfig {
    int vocab_size = 151936;
    int hidden_size = 896;
    int intermediate_size = 4864;
    int num_hidden_layers = 24;
    int num_attention_heads = 14;
    int num_key_value_heads = 2;
    int max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;

    // Keep the original aggregate-initialization prefix source-compatible.
    std::string model_type = "qwen2";
    int head_dim = 0; // zero: Qwen2 derives it; Qwen3 defaults to 128, Qwen3.5 to 256
    bool attention_bias = true;
    bool tie_word_embeddings = true;

    static QwenConfig from_pretrained(const std::string& config_path);
    int attention_head_dim() const;
    std::vector<std::string> default_lora_targets() const;
    bool hybrid() const { return model_type == "qwen3_5_text"; }
    bool uses_qk_norm() const { return model_type == "qwen3" || hybrid(); }
    std::string checkpoint_prefix = "model.";
    std::vector<std::string> layer_types{};
    int linear_num_key_heads = 16, linear_num_value_heads = 32;
    int linear_key_head_dim = 128, linear_value_head_dim = 128, linear_conv_kernel_dim = 4;
    float partial_rotary_factor = 0.25f;
    bool linear_layer(size_t i) const {
        return hybrid() && layer_types.at(i) == "linear_attention";
    }
    void validate() const;
};

struct QwenBlock {
    TensorPtr input_norm_weight;
    TensorPtr post_norm_weight;
    TensorPtr q_norm_weight, k_norm_weight;

    TensorPtr q_proj_weight, q_proj_bias;
    TensorPtr k_proj_weight, k_proj_bias;
    TensorPtr v_proj_weight, v_proj_bias;
    TensorPtr o_proj_weight, o_proj_bias;

    TensorPtr gate_proj_weight;
    TensorPtr up_proj_weight;
    TensorPtr down_proj_weight;

    std::unique_ptr<LoRALinear> q_lin;
    std::unique_ptr<LoRALinear> k_lin;
    std::unique_ptr<LoRALinear> v_lin;
    std::unique_ptr<LoRALinear> o_lin;
    std::unique_ptr<LoRALinear> gate_lin, up_lin, down_lin;

    TensorPtr in_proj_qkv_weight, in_proj_z_weight, in_proj_a_weight, in_proj_b_weight,
        out_proj_weight;
    TensorPtr conv_weight, log_decay, dt_bias, gated_norm_weight;
    std::unique_ptr<LoRALinear> in_qkv_lin, in_z_lin, in_a_lin, in_b_lin, out_lin;
    bool lora_initialized = false;
};

class QwenModel {
  public:
    explicit QwenModel(const QwenConfig& cfg);
    ~QwenModel() = default;

    TensorPtr forward_hidden(const TensorPtr& input_ids, const TensorPtr& attention_mask = nullptr);
    TensorPtr lm_head(const TensorPtr& hidden);
    TensorPtr forward(const TensorPtr& input_ids, const TensorPtr& attention_mask = nullptr);
    const TensorPtr& embedding_weight() const { return embed_tokens_; }
    const TensorPtr& output_weight() const {
        return config_.tie_word_embeddings ? embed_tokens_ : lm_head_weight_;
    }

    // Complete, shape-checked checkpoint loading; missing weights are errors.
    void load_pretrained(const std::string& model_dir, const SafeTensorsLoadOptions& options);

    /**
     * @brief Initialize LoRA
     * @param rank LoRA rank
     * @param alpha LoRA alpha (scaling factor = alpha / rank)
     * @param dropout LoRA dropout
     * @param qv_only If true, attach only q_proj and v_proj (default false = all
     * attention mixer projections)
     */
    void init_lora(int rank = 8, float alpha = 16.0f, float dropout = 0.05f, bool qv_only = false,
                   uint64_t seed = 42);
    void init_lora(int rank, float alpha, float dropout, const std::vector<std::string>& targets,
                   uint64_t seed = 42);
    std::unordered_map<std::string, TensorPtr> named_lora_parameters() const;
    void save_lora_adapter(const std::string& directory) const;
    void load_lora_adapter(const std::string& directory);
    void set_training(bool training);
    int lora_rank() const { return lora_rank_; }
    float lora_alpha() const { return lora_alpha_; }
    float lora_dropout() const { return lora_dropout_; }
    const std::vector<std::string>& lora_targets() const { return lora_targets_; }
    std::vector<TensorPtr> get_lora_parameters() const;
    std::vector<TensorPtr> parameters() const;
    void freeze_base();

    void set_memory_efficient_attention(bool enabled) { memory_efficient_attention_ = enabled; }
    void set_activation_checkpointing(bool enabled);
    void enable_parameter_sharding(const sharding::ShardConfig& config);
    sharding::ShardStats sharding_stats() const {
        return offload_ ? offload_->stats() : sharding::ShardStats{};
    }

    void assign_weight(const std::string& key, const TensorPtr& tensor);
    const QwenConfig& config() const { return config_; }

  private:
    QwenConfig config_;
    TensorPtr embed_tokens_;      // [vocab, hidden]
    TensorPtr final_norm_weight_; // [hidden]
    TensorPtr lm_head_weight_;    // [vocab, hidden], only for untied heads

    std::vector<QwenBlock> layers_;
    bool memory_efficient_attention_ = false;
    bool activation_checkpointing_ = false;
    bool training_ = true;
    int lora_rank_ = 0;
    float lora_alpha_ = 0.0f, lora_dropout_ = 0.0f;
    std::vector<std::string> lora_targets_;
    struct LinearModule {
        const char* name;
        const char* scope;
        TensorPtr* weight;
        TensorPtr* bias;
        std::unique_ptr<LoRALinear>* adapter;
    };
    std::vector<LinearModule> linear_modules(QwenBlock& block);
    std::unordered_map<std::string, TensorPtr*> weight_slots();
    std::vector<std::string> available_targets();
    TensorPtr norm(const TensorPtr& x, const TensorPtr& weight);
    TensorPtr linear_attention(const TensorPtr& x, QwenBlock& block, const TensorPtr& mask);
    std::unique_ptr<FrozenModelOffload> offload_;

    TensorPtr embedding_lookup(const TensorPtr& weight, const TensorPtr& indices);
    TensorPtr build_causal_mask(int seq_len);
    TensorPtr attention(const TensorPtr& x, QwenBlock& blk, const TensorPtr& causal_mask,
                        const TensorPtr& pad_mask, int64_t seq_len);
    TensorPtr mlp(const TensorPtr& x, QwenBlock& blk);
};

} // namespace ops
