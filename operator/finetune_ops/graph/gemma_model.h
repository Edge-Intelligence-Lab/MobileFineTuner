#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace ops {

/**
 * @brief Text-only Gemma3 270M config (aligned with HuggingFace Gemma3TextConfig)
 */
struct GemmaTextConfig {
    int vocab_size = 262144;
    int hidden_size = 640;
    int intermediate_size = 2048;
    int num_hidden_layers = 18;
    int num_attention_heads = 4;
    int num_key_value_heads = 1;
    int head_dim = 256;
    int max_position_embeddings = 32768;
    int sliding_window = 512;
    bool attention_bias = false;
    bool use_bidirectional_attention = false;
    bool use_cache = true;

    float attention_dropout = 0.0f;
    float rms_norm_eps = 1e-6f;
    float query_pre_attn_scalar = 256.0f;
    float attn_logit_softcapping = 0.0f;
    float final_logit_softcapping = 0.0f;
    float rope_theta = 1000000.0f;
    float rope_local_base_freq = 10000.0f;

    std::string hidden_activation = "gelu_pytorch_tanh";
    std::vector<std::string> layer_types;

    static GemmaTextConfig from_pretrained(const std::string& model_dir);
};

struct GemmaBlockWeights {
    TensorPtr input_layernorm_weight;
    TensorPtr post_attention_layernorm_weight;
    TensorPtr pre_feedforward_layernorm_weight;
    TensorPtr post_feedforward_layernorm_weight;

    TensorPtr q_proj_weight;
    TensorPtr k_proj_weight;
    TensorPtr v_proj_weight;
    TensorPtr o_proj_weight;

    TensorPtr q_norm_weight;
    TensorPtr k_norm_weight;

    TensorPtr gate_proj_weight;
    TensorPtr up_proj_weight;
    TensorPtr down_proj_weight;

    std::unique_ptr<LoRALinear> q_proj_lora;
    std::unique_ptr<LoRALinear> k_proj_lora;
    std::unique_ptr<LoRALinear> v_proj_lora;
    std::unique_ptr<LoRALinear> o_proj_lora;
    std::unique_ptr<LoRALinear> gate_proj_lora;
    std::unique_ptr<LoRALinear> up_proj_lora;
    std::unique_ptr<LoRALinear> down_proj_lora;

    bool lora_initialized = false;
};

struct EmbeddingDumpRequest {
    bool active = false;
    int target_step = 0;
    std::string output_dir;
    bool fulfilled = false;
};

class GemmaModel {
public:
    explicit GemmaModel(const GemmaTextConfig& config);
    ~GemmaModel() = default;

    TensorPtr forward(const TensorPtr& input_ids,
                      const TensorPtr& attention_mask = nullptr);

    void assign_weight(const std::string& key, const TensorPtr& tensor);

    const GemmaTextConfig& config() const { return config_; }

    void init_lora_modules();
    GemmaBlockWeights& get_block(int i);
    const GemmaBlockWeights& get_block(int i) const;
    std::vector<TensorPtr> get_lora_parameters() const;
    void merge_lora();
    void unmerge_lora();

    void request_embedding_dump(int step, const std::string& output_dir);
    
    // Alignment debugging: enable/disable intermediate activation dumps
    void enable_debug_dump(const std::string& dir, const std::vector<int>& layers);
    void disable_debug_dump();
    bool debug_enabled() const { return debug_.enabled; }
    const std::unordered_map<std::string, TensorPtr>& debug_tensors() const { return debug_.tensors; }
    void set_debug_retain_grads(bool retain) { debug_.retain_grads = retain; }
    // Numeric gradient perturbation control (debug only)
    void set_numeric_perturb(bool enable, const std::string& name, int64_t index, float eps);
    // Debug helper: dump layer RMSNorm weights for inspection
    void dump_layer_norm_weights(int layer, const std::string& dir) const;

private:
    GemmaTextConfig config_;
    TensorPtr embed_weight_;
    TensorPtr norm_weight_;
    TensorPtr lm_head_weight_;
    std::vector<GemmaBlockWeights> blocks_;
    bool lm_head_initialized_ = false;
    EmbeddingDumpRequest dump_request_;
    
    struct DebugConfig {
        bool enabled = false;
        std::string dir;
        std::unordered_set<int> layers;
        mutable std::unordered_map<std::string, TensorPtr> tensors;
        bool numeric_enabled = false;
        std::string numeric_name;
        int64_t numeric_index = -1;  // flatten index in target tensor
        float numeric_eps = 1e-3f;
        bool retain_grads = true;
    };
    mutable DebugConfig debug_;
    bool need_dump_layer(int idx) const;
    void dump_tensor(const TensorPtr& t, const std::string& name) const;

    TensorPtr build_causal_mask(int seq_len) const;
    TensorPtr build_sliding_mask(int seq_len) const;
    TensorPtr build_padding_mask(const TensorPtr& attention_mask) const;

    struct RotaryCache {
        TensorPtr cos;
        TensorPtr sin;
    };

    RotaryCache build_rotary_embeddings(int batch, int seq_len, float theta) const;

    TensorPtr embedding_lookup(const TensorPtr& indices) const;
    TensorPtr forward_block(const TensorPtr& hidden_states,
                            GemmaBlockWeights& block,
                            const TensorPtr& position_cos,
                            const TensorPtr& position_sin,
                            const TensorPtr& pad_mask,
                            const TensorPtr& base_mask,
                            float rope_theta) const;

    TensorPtr apply_attention(const TensorPtr& x,
                              GemmaBlockWeights& block,
                              const TensorPtr& position_cos,
                              const TensorPtr& position_sin,
                              const TensorPtr& pad_mask,
                              const TensorPtr& base_mask,
                              float rope_theta,
                              int dbg_layer = -1) const;

    TensorPtr apply_mlp(const TensorPtr& x,
                        GemmaBlockWeights& block,
                        int dbg_layer = -1) const;

    void maybe_dump_embedding(const TensorPtr& hidden_states, int batch, int seq_len, int hidden_dim);
};

}  // namespace ops
