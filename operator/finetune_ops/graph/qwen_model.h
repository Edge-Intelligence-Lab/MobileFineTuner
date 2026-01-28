/**
 * @file qwen_model.h
 * @brief Minimal Qwen2.5-0.5B C++ implementation (RMSNorm + RoPE + GQA + SwiGLU)
 *
 * Goal: CPU + BLAS training/alignment, supports attention-only LoRA (q/k/v/o).
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops {

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

    static QwenConfig from_pretrained(const std::string& config_path);
};

struct QwenBlock {
    TensorPtr input_norm_weight;
    TensorPtr post_norm_weight;

    TensorPtr q_proj_weight, q_proj_bias;
    TensorPtr k_proj_weight, k_proj_bias;
    TensorPtr v_proj_weight, v_proj_bias;
    TensorPtr o_proj_weight;

    TensorPtr gate_proj_weight;
    TensorPtr up_proj_weight;
    TensorPtr down_proj_weight;

    std::unique_ptr<LoRALinear> q_lin;
    std::unique_ptr<LoRALinear> k_lin;
    std::unique_ptr<LoRALinear> v_lin;
    std::unique_ptr<LoRALinear> o_lin;

    bool lora_initialized = false;
};

class QwenModel {
public:
    explicit QwenModel(const QwenConfig& cfg);
    ~QwenModel() = default;

    TensorPtr forward(const TensorPtr& input_ids, const TensorPtr& attention_mask = nullptr);

    /**
     * @brief Initialize LoRA
     * @param rank LoRA rank
     * @param alpha LoRA alpha (scaling factor = alpha / rank)
     * @param dropout LoRA dropout
     * @param qv_only If true, attach only q_proj and v_proj (default false = q/k/v/o)
     */
    void init_lora(int rank = 8, float alpha = 16.0f, float dropout = 0.05f, bool qv_only = false);
    std::vector<TensorPtr> get_lora_parameters() const;
    void freeze_base();

    void assign_weight(const std::string& key, const TensorPtr& tensor);
    const QwenConfig& config() const { return config_; }

private:
    QwenConfig config_;
    TensorPtr embed_tokens_;      // [vocab, hidden]
    TensorPtr final_norm_weight_; // [hidden]

    std::vector<QwenBlock> layers_;

    TensorPtr embedding_lookup(const TensorPtr& weight, const TensorPtr& indices);
    TensorPtr build_causal_mask(int seq_len);
    TensorPtr attention(const TensorPtr& x, QwenBlock& blk,
                        const TensorPtr& causal_mask,
                        const TensorPtr& pad_mask, int64_t seq_len);
    TensorPtr mlp(const TensorPtr& x, QwenBlock& blk);
};

}  // namespace ops

