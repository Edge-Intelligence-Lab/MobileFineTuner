/**
 * @file gpt2_model.h
 * @brief Lightweight GPT-2 wrapper (assembles gpt2_components, supports tie-weights and LoRA injection)
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace ops {

/**
 * @brief GPT-2 Transformer Block weights (includes LoRALinear modules)
 */
struct GPT2BlockWeights {
    // LayerNorm
    TensorPtr ln_1_weight, ln_1_bias;
    TensorPtr ln_2_weight, ln_2_bias;
    
    // Attention fused weights (for initialization)
    TensorPtr attn_qkv_weight;   // [C, 3C]
    TensorPtr attn_qkv_bias;     // [3C]
    TensorPtr attn_proj_weight;  // [C, C]
    TensorPtr attn_proj_bias;    // [C]
    
    // MLP weights
    TensorPtr mlp_fc_in_weight;   // [C, 4C]
    TensorPtr mlp_fc_in_bias;     // [4C]
    TensorPtr mlp_fc_out_weight;  // [4C, C]
    TensorPtr mlp_fc_out_bias;    // [C]
    
    // LoRA-augmented linear layers
    std::unique_ptr<LoRALinear> qkv_lin;
    std::unique_ptr<LoRALinear> proj_lin;
    std::unique_ptr<LoRALinear> fc_in_lin;
    std::unique_ptr<LoRALinear> fc_out_lin;
    
    bool lora_initialized = false;
};

/**
 * @brief GPT-2 configuration (aligned with HuggingFace)
 */
struct GPT2Config {
    int vocab_size = 50257;
    int n_positions = 1024;
    int n_embd = 768;
    int n_layer = 12;
    int n_head = 12;
    float layernorm_eps = 1e-5f;      // Same as HF
    bool tie_word_embeddings = true;  // lm_head ↔ wte
    bool use_cache = false;           // Disable KV cache during training
    
    // Memory optimizations
    bool use_memory_efficient_attention = true;   // Streamed softmax (recommended)
    bool use_bf16_activations = false;            // Downcast activations to BF16 when needed
    
    // Load from config.json (future work)
    static GPT2Config from_pretrained(const std::string& config_path);
};

/**
 * @brief GPT-2 model (lightweight wrapper)
 * 
 * Assembles Embedding + TransformerBlock + LayerNorm + lm_head
 * Supports:
 * - tie_weights (lm_head ↔ wte)
 * - assign_weight (for safetensors_loader to populate)
 * - forward inference (input_ids + attention_mask → logits)
 */
class GPT2Model {
public:
    explicit GPT2Model(const GPT2Config& config);
    ~GPT2Model() = default;
    
    /**
     * @brief Forward pass
     * @param input_ids [batch, seq_len] int32
     * @param attention_mask [batch, seq_len] int32/float (1=valid, 0=pad)
     * @return logits [batch, seq_len, vocab_size]
     */
    TensorPtr forward(const TensorPtr& input_ids,
                     const TensorPtr& attention_mask = nullptr);
    
    /**
     * @brief Tie lm_head.weight ↔ wte.weight (shared memory)
     */
    void tie_weights();
    
    /**
     * @brief Receive weights from safetensors_loader
     * @param key internal key (e.g., "wte.weight", "blocks.0.ln_1.weight")
     * @param tensor weight tensor
     */
    void assign_weight(const std::string& key, const TensorPtr& tensor);
    
    /**
     * @brief Fetch all parameters (for optimizer)
     */
    std::vector<TensorPtr> parameters();
    
    /**
     * @brief Fetch trainable parameters (after LoRA injection, only LoRA params)
     */
    std::vector<TensorPtr> trainable_parameters();
    
    /**
     * @brief Freeze all parameters except LoRA
     */
    void freeze_base_parameters();
    
    /**
     * @brief Print model info
     */
    void print_model_info() const;
    
    const GPT2Config& config() const { return config_; }

    // ========================= Getters for LoRA binding =========================
    // Return writable pointers (TensorPtr*) so the LoRA injector can attach to layer params
    // i ∈ [0, config_.n_layer)
    std::pair<TensorPtr*, TensorPtr*> attn_qkv_params(int i);
    std::pair<TensorPtr*, TensorPtr*> attn_proj_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_in_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_out_params(int i);
    
    /**
     * @brief Access block (for LoRA injector)
     */
    GPT2BlockWeights& get_block(int i) { return blocks_[i]; }
    const GPT2BlockWeights& get_block(int i) const { return blocks_[i]; }
    
    /**
     * @brief Collect all LoRA trainable parameters
     */
    std::vector<TensorPtr> get_lora_parameters();
    
    /**
     * @brief Initialize LoRA modules (call after loading weights)
     */
    void init_lora_modules();

private:
    GPT2Config config_;
    
    // Embeddings
    TensorPtr wte_weight_;  // [vocab_size, n_embd]
    TensorPtr wpe_weight_;  // [n_positions, n_embd]
    
    // Transformer blocks (using externally defined GPT2BlockWeights)
    std::vector<GPT2BlockWeights> blocks_;
    
    // Final LayerNorm
    TensorPtr ln_f_weight_;
    TensorPtr ln_f_bias_;
    
    // lm_head (tied to wte_weight_, not stored separately)
    bool weights_tied_ = false;
    
    // 内部工具
    TensorPtr build_causal_mask(int seq_len);
    TensorPtr build_padding_mask(const TensorPtr& attention_mask);
    TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias);
    TensorPtr gelu_new(const TensorPtr& x);
    TensorPtr embedding_lookup(const TensorPtr& weight, const TensorPtr& indices);
    TensorPtr forward_block(const TensorPtr& x, int block_idx,
                           const TensorPtr& causal_mask,
                           const TensorPtr& pad_mask);
};

}  // namespace ops

