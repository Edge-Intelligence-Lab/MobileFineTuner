/**
 * @file gpt2_model.cpp
 * @brief GPT-2 model implementation
 */

#include "gpt2_model.h"
#include "../core/ops.h"
#include "../core/memory_efficient_attention.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <regex>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace ops {

// ----------------------------------------------------------------------------
// GPT2Config loader from HuggingFace-style config.json
// ----------------------------------------------------------------------------
namespace {
static std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int extract_int_field(const std::string& content, const std::string& key, int def) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*(-?\d+))";
    std::regex pattern(pattern_str);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return def;
}

static float extract_float_field(const std::string& content, const std::string& key, float def) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*(-?\d+(\.\d+)?([eE][-+]?\d+)?))";
    std::regex pattern(pattern_str);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return std::stof(match[1].str());
    }
    return def;
}
}  // namespace

GPT2Config GPT2Config::from_pretrained(const std::string& path_like) {
    GPT2Config cfg;
    namespace fs = std::filesystem;
    fs::path p(path_like);
    fs::path cfg_path = p;
    if (fs::is_directory(p)) {
        cfg_path = p / "config.json";
    }
    const std::string content = read_text_file(cfg_path.string());
    // HF GPT-2 fields
    cfg.vocab_size = extract_int_field(content, "vocab_size", cfg.vocab_size);
    // n_positions or n_ctx
    int n_pos = extract_int_field(content, "n_positions", -1);
    if (n_pos < 0) n_pos = extract_int_field(content, "n_ctx", cfg.n_positions);
    cfg.n_positions = n_pos;
    cfg.n_embd = extract_int_field(content, "n_embd", cfg.n_embd);
    cfg.n_layer = extract_int_field(content, "n_layer", cfg.n_layer);
    cfg.n_head  = extract_int_field(content, "n_head",  cfg.n_head);
    cfg.layernorm_eps = extract_float_field(content, "layer_norm_epsilon", cfg.layernorm_eps);
    // Defaults for training
    cfg.tie_word_embeddings = true;
    cfg.use_cache = false;
    return cfg;
}

// Compile-time switch for verbose internal debug prints in forward pass.
// Default OFF. Enable by passing -DENABLE_GPT2_DEBUG_PRINT=ON to CMake.
#if defined(ENABLE_GPT2_DEBUG_PRINT)
static constexpr bool GPT2_DEBUG_PRINT = true;
#else
static constexpr bool GPT2_DEBUG_PRINT = false;
#endif

namespace {

enum class DumpDType { kFloat32, kInt32 };

// Minimal .npy writer for float32/int32, row-major, C-order
bool save_npy(const std::string& path,
              const void* data,
              const std::vector<int64_t>& shape,
              DumpDType dtype) {
    std::string descr = (dtype == DumpDType::kFloat32) ? "<f4" : "<i4";
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1) shape_str += ",";
        if (i + 1 < shape.size()) shape_str += ", ";
    }
    shape_str += ")";
    std::string header_dict = "{'descr': '" + descr +
        "', 'fortran_order': False, 'shape': " + shape_str + ", }";

    std::string magic = "\x93NUMPY";
    uint8_t ver_major = 1, ver_minor = 0;
    size_t header_len = header_dict.size() + 1;  // newline
    size_t preamble = magic.size() + 2 + 2;
    size_t padding = 16 - ((preamble + header_len) % 16);
    if (padding == 16) padding = 0;
    header_dict += std::string(padding, ' ');
    header_dict.push_back('\n');
    uint16_t header_size_le = static_cast<uint16_t>(header_dict.size());

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(magic.data(), magic.size());
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));
    out.write(reinterpret_cast<const char*>(&header_size_le), sizeof(header_size_le));
    out.write(header_dict.data(), header_dict.size());

    size_t count = 1;
    for (auto d : shape) count *= static_cast<size_t>(d);
    size_t elem_size = 4;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * elem_size));
    return true;
}

}  // namespace

// ============================================================================
// Construction and initialization
// ============================================================================

GPT2Model::GPT2Model(const GPT2Config& config)
    : config_(config), weights_tied_(false) {
    
    // Initialize embeddings (placeholders; weights filled by loader)
    wte_weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{config_.vocab_size, config_.n_embd}, kFloat32, kCPU);
    wpe_weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{config_.n_positions, config_.n_embd}, kFloat32, kCPU);
    
    // Initialize transformer blocks
    blocks_.resize(config_.n_layer);
    for (int i = 0; i < config_.n_layer; ++i) {
        auto& block = blocks_[i];
        
        // LayerNorm 1
        block.ln_1_weight = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
        block.ln_1_bias = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
        
        // Attention QKV (fused)
        block.attn_qkv_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.n_embd, 3 * config_.n_embd}, kFloat32, kCPU);
        block.attn_qkv_bias = std::make_shared<Tensor>(
            std::vector<int64_t>{3 * config_.n_embd}, kFloat32, kCPU);
        
        // Attention proj
        block.attn_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.n_embd, config_.n_embd}, kFloat32, kCPU);
        block.attn_proj_bias = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
        
        // LayerNorm 2
        block.ln_2_weight = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
        block.ln_2_bias = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
        
        // MLP
        block.mlp_fc_in_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.n_embd, 4 * config_.n_embd}, kFloat32, kCPU);
        block.mlp_fc_in_bias = std::make_shared<Tensor>(
            std::vector<int64_t>{4 * config_.n_embd}, kFloat32, kCPU);
        
        block.mlp_fc_out_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{4 * config_.n_embd, config_.n_embd}, kFloat32, kCPU);
        block.mlp_fc_out_bias = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
    }
    
    // Final LayerNorm
    ln_f_weight_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
    ln_f_bias_ = std::make_shared<Tensor>(std::vector<int64_t>{config_.n_embd}, kFloat32, kCPU);
    
    std::cout << "[GPT2Model] Initialized with " << config_.n_layer << " layers, "
              << config_.n_embd << " hidden, " << config_.n_head << " heads" << std::endl;
}

// ============================================================================
// Weight management
// ============================================================================

void GPT2Model::tie_weights() {
    // Share lm_head weight with wte (no separate allocation)
    // Forward pass uses wte_weight_ directly for the final projection
    weights_tied_ = true;
    std::cout << "[GPT2Model] Tied lm_head.weight ↔ wte.weight" << std::endl;
}

void GPT2Model::init_lora_modules() {
    // Initialize LoRALinear modules for each block (referencing loaded weights)
    for (auto& block : blocks_) {
        if (block.lora_initialized) continue;
        
        // Create LoRALinear modules (referencing base weights)
        block.qkv_lin = std::make_unique<LoRALinear>(
            block.attn_qkv_weight, block.attn_qkv_bias);
        block.proj_lin = std::make_unique<LoRALinear>(
            block.attn_proj_weight, block.attn_proj_bias);
        block.fc_in_lin = std::make_unique<LoRALinear>(
            block.mlp_fc_in_weight, block.mlp_fc_in_bias);
        block.fc_out_lin = std::make_unique<LoRALinear>(
            block.mlp_fc_out_weight, block.mlp_fc_out_bias);
        
        block.lora_initialized = true;
    }
}

std::vector<TensorPtr> GPT2Model::get_lora_parameters() {
    std::vector<TensorPtr> params;
    for (const auto& block : blocks_) {
        if (!block.lora_initialized) continue;
        
        auto add_params = [&](const std::unique_ptr<LoRALinear>& lin) {
            if (lin) {
                auto ps = lin->trainable_parameters();
                params.insert(params.end(), ps.begin(), ps.end());
            }
        };
        
        add_params(block.qkv_lin);
        add_params(block.proj_lin);
        add_params(block.fc_in_lin);
        add_params(block.fc_out_lin);
    }
    return params;
}

void GPT2Model::assign_weight(const std::string& key, const TensorPtr& tensor) {
    // 按内部键名分发到对应权重
    if (key == "wte.weight") {
        wte_weight_ = tensor;
    } else if (key == "wpe.weight") {
        wpe_weight_ = tensor;
    } else if (key == "ln_f.weight") {
        ln_f_weight_ = tensor;
    } else if (key == "ln_f.bias") {
        ln_f_bias_ = tensor;
    } else {
        // 解析 blocks.N.xxx
        std::regex block_pattern(R"(blocks\.(\d+)\.(.+))");
        std::smatch match;
        if (std::regex_match(key, match, block_pattern)) {
            int block_idx = std::stoi(match[1].str());
            std::string sub_key = match[2].str();
            
            if (block_idx < 0 || block_idx >= config_.n_layer) {
                throw std::runtime_error("Invalid block index: " + std::to_string(block_idx));
            }
            
            auto& block = blocks_[block_idx];
            
            if (sub_key == "ln_1.weight") block.ln_1_weight = tensor;
            else if (sub_key == "ln_1.bias") block.ln_1_bias = tensor;
            else if (sub_key == "attn.qkv.weight") block.attn_qkv_weight = tensor;
            else if (sub_key == "attn.qkv.bias") block.attn_qkv_bias = tensor;
            else if (sub_key == "attn.proj.weight") block.attn_proj_weight = tensor;
            else if (sub_key == "attn.proj.bias") block.attn_proj_bias = tensor;
            else if (sub_key == "ln_2.weight") block.ln_2_weight = tensor;
            else if (sub_key == "ln_2.bias") block.ln_2_bias = tensor;
            else if (sub_key == "mlp.fc_in.weight") block.mlp_fc_in_weight = tensor;
            else if (sub_key == "mlp.fc_in.bias") block.mlp_fc_in_bias = tensor;
            else if (sub_key == "mlp.fc_out.weight") block.mlp_fc_out_weight = tensor;
            else if (sub_key == "mlp.fc_out.bias") block.mlp_fc_out_bias = tensor;
            else {
                std::cerr << "[WARN] Unknown sub_key: " << sub_key << std::endl;
            }
        } else {
            std::cerr << "[WARN] Unknown key: " << key << std::endl;
        }
    }
}

// =========================================================================
// Getters (needed for LoRA binding)
// =========================================================================

std::pair<TensorPtr*, TensorPtr*> GPT2Model::attn_qkv_params(int i) {
    if (i < 0 || i >= config_.n_layer) throw std::out_of_range("attn_qkv_params: layer index");
    auto& b = blocks_[i];
    return { &b.attn_qkv_weight, &b.attn_qkv_bias };
}

std::pair<TensorPtr*, TensorPtr*> GPT2Model::attn_proj_params(int i) {
    if (i < 0 || i >= config_.n_layer) throw std::out_of_range("attn_proj_params: layer index");
    auto& b = blocks_[i];
    return { &b.attn_proj_weight, &b.attn_proj_bias };
}

std::pair<TensorPtr*, TensorPtr*> GPT2Model::mlp_fc_in_params(int i) {
    if (i < 0 || i >= config_.n_layer) throw std::out_of_range("mlp_fc_in_params: layer index");
    auto& b = blocks_[i];
    return { &b.mlp_fc_in_weight, &b.mlp_fc_in_bias };
}

std::pair<TensorPtr*, TensorPtr*> GPT2Model::mlp_fc_out_params(int i) {
    if (i < 0 || i >= config_.n_layer) throw std::out_of_range("mlp_fc_out_params: layer index");
    auto& b = blocks_[i];
    return { &b.mlp_fc_out_weight, &b.mlp_fc_out_bias };
}

// ============================================================================
// Forward pass (core)
// ============================================================================

TensorPtr GPT2Model::forward(const TensorPtr& input_ids,
                             const TensorPtr& attention_mask) {
    const auto& shape = input_ids->shape();
    int64_t batch = shape[0];
    int64_t seq_len = shape[1];
    static bool dumped_once = false;
    const char* dump_dir_c = std::getenv("GPT2_ALIGN_DUMP_DIR");
    std::string dump_dir = dump_dir_c ? std::string(dump_dir_c) : std::string();
    
    if (seq_len > config_.n_positions) {
        throw std::runtime_error("seq_len exceeds n_positions");
    }
    
    // 1. Embeddings: x = wte[input_ids] + wpe[0:seq_len]
    TensorPtr x = embedding_lookup(wte_weight_, input_ids);
    if (!dumped_once && !dump_dir.empty()) {
        // input_ids
        save_npy(dump_dir + "/input_ids.npy",
                 input_ids->data<int32_t>(),
                 input_ids->shape(),
                 DumpDType::kInt32);
        // embedding lookup result
        save_npy(dump_dir + "/embeddings.npy",
                 x->data<float>(),
                 x->shape(),
                 DumpDType::kFloat32);
    }
    
    // Positional embedding (manual slice wpe[0:seq_len])
    auto pos_emb = zeros({seq_len, config_.n_embd}, kFloat32, kCPU);
    const float* wpe_data = wpe_weight_->data<float>();
    float* pos_data = pos_emb->data<float>();
    std::memcpy(pos_data, wpe_data, seq_len * config_.n_embd * sizeof(float));
    if (!dumped_once && !dump_dir.empty()) {
        save_npy(dump_dir + "/pos_emb.npy",
                 pos_emb->data<float>(),
                 pos_emb->shape(),
                 DumpDType::kFloat32);
    }
    
    // Broadcast add: x[B,S,C] + pos_emb[S,C] (auto broadcast to [B,S,C])
    const float* pos_emb_data = pos_emb->data<float>();
    float* x_data = x->data<float>();
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t c = 0; c < config_.n_embd; ++c) {
                x_data[(b * seq_len + s) * config_.n_embd + c] += pos_emb_data[s * config_.n_embd + c];
            }
        }
    }
    if (!dumped_once && !dump_dir.empty()) {
        // attention mask (if provided)
        if (attention_mask) {
            save_npy(dump_dir + "/attention_mask.npy",
                     attention_mask->data<float>(),
                     attention_mask->shape(),
                     DumpDType::kFloat32);
        }
        // embeddings + pos
        save_npy(dump_dir + "/emb_plus_pos.npy",
                 x->data<float>(),
                 x->shape(),
                 DumpDType::kFloat32);
        dumped_once = true;
    }
    
    
    // 2. Build masks
    TensorPtr causal = build_causal_mask(seq_len);
    TensorPtr pad_mask = attention_mask ? build_padding_mask(attention_mask) : nullptr;
    
    // 3. Transformer blocks
    for (int i = 0; i < config_.n_layer; ++i) {
        x = forward_block(x, i, causal, pad_mask);
    }
    
    // 4. Final LayerNorm
    x = layer_norm(x, ln_f_weight_, ln_f_bias_);
    
    // 5. lm_head (tied to wte; no transpose because wte is [V,C] used as embedding)
    // Linear weights are [in,out], so wte is [V,C]=[out,in] for embedding.
    // lm_head needs [C,V], hence transpose.
    auto wte_t = transpose(wte_weight_, 0, 1);  // [C,V]
    TensorPtr logits = matmul(x, wte_t);  // [B,S,C] @ [C,V] → [B,S,V]
    
    return logits;
}

// ============================================================================
// Internal utilities
// ============================================================================

TensorPtr GPT2Model::build_causal_mask(int seq_len) {
    // [seq_len, seq_len]; lower triangle = 0, upper triangle = -inf
    // Added to attention scores before softmax
    auto mask = full({seq_len, seq_len}, 0.0f, kFloat32, kCPU);
    float* data = mask->data<float>();
    
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = i + 1; j < seq_len; ++j) {
            data[i * seq_len + j] = -1e10f;  // Set upper triangle to a large negative value
        }
    }
    
    return mask;
}

TensorPtr GPT2Model::build_padding_mask(const TensorPtr& attention_mask) {
    // attention_mask: [B, S] (1=keep, 0=pad)
    // Return shape [B, 1, 1, S]; pad positions = -inf, valid = 0
    const auto& shape = attention_mask->shape();
    int64_t batch = shape[0];
    int64_t seq_len = shape[1];
    
    auto mask = zeros({batch, 1, 1, seq_len}, kFloat32, kCPU);
    float* mask_data = mask->data<float>();
    const float* attn_data = attention_mask->data<float>();
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            if (attn_data[b * seq_len + s] == 0.0f) {
                mask_data[b * seq_len + s] = -1e10f;
            }
        }
    }
    
    return mask;
}

TensorPtr GPT2Model::layer_norm(const TensorPtr& x,
                                const TensorPtr& weight,
                                const TensorPtr& bias) {
    return ops::layer_norm(x, weight, bias, config_.layernorm_eps);
}

TensorPtr GPT2Model::gelu_new(const TensorPtr& x) {
    // GELU new (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return ops::gelu(x);  // Assumes ops::gelu implements the tanh variant
}

// ============================================================================
// Helper functions (TODO: these basic ops require implementations)
// ============================================================================

TensorPtr GPT2Model::embedding_lookup(const TensorPtr& weight, const TensorPtr& indices) {
    // weight: [vocab, embd]
    // indices: [B, S] int32
    // Return: [B, S, embd]
    const auto& idx_shape = indices->shape();
    int64_t batch = idx_shape[0];
    int64_t seq_len = idx_shape[1];
    int64_t embd = weight->shape()[1];
    
    auto result = zeros({batch, seq_len, embd}, kFloat32, kCPU);
    const float* weight_data = weight->data<float>();
    const int32_t* idx_data = indices->data<int32_t>();
    float* result_data = result->data<float>();
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            int32_t idx = idx_data[b * seq_len + s];
            if (idx < 0 || idx >= weight->shape()[0]) {
                throw std::runtime_error("Index out of range: " + std::to_string(idx));
            }
            
            const float* emb_vec = weight_data + idx * embd;
            float* out_vec = result_data + (b * seq_len + s) * embd;
            std::memcpy(out_vec, emb_vec, embd * sizeof(float));
        }
    }
    
    return result;
}

TensorPtr GPT2Model::forward_block(const TensorPtr& x_in,
                                   int block_idx,
                                   const TensorPtr& causal_mask,
                                   const TensorPtr& pad_mask) {
    const auto& w = blocks_[block_idx];
    
    const int64_t B = x_in->shape()[0];
    const int64_t S = x_in->shape()[1];
    const int64_t C = x_in->shape()[2];
    const int64_t n_head = config_.n_head;
    const int64_t Hd = C / n_head;  // head_dim = 64 for GPT-2
    
    // ========== Attention ==========
    TensorPtr residual = x_in;
    TensorPtr x = layer_norm(x_in, w.ln_1_weight, w.ln_1_bias);
    
    // 🔍 DEBUG: Layer 11 ln_1 output and QKV weights
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        const float* ln1_data = x->data<float>();
        double sum = 0.0, sum_sq = 0.0;
        for (int64_t i = 0; i < x->numel(); ++i) {
            sum += ln1_data[i];
            sum_sq += ln1_data[i] * ln1_data[i];
        }
        float mean = sum / x->numel();
        float std = std::sqrt(sum_sq / x->numel() - mean * mean);
        printf("[DEBUG] Layer 11 ln_1 output: mean=%10.6f, std=%10.6f\n", mean, std);
        
        // QKV weight statistics
        const float* qkv_w_data = w.attn_qkv_weight->data<float>();
        double w_sum = 0.0, w_sum_sq = 0.0;
        for (int64_t i = 0; i < w.attn_qkv_weight->numel(); ++i) {
            w_sum += qkv_w_data[i];
            w_sum_sq += qkv_w_data[i] * qkv_w_data[i];
        }
        float w_mean = w_sum / w.attn_qkv_weight->numel();
        float w_std = std::sqrt(w_sum_sq / w.attn_qkv_weight->numel() - w_mean * w_mean);
        printf("[DEBUG] Layer 11 attn_qkv_weight: mean=%10.6f, std=%10.6f, shape=[%lld,%lld]\n", 
               w_mean, w_std, (long long)w.attn_qkv_weight->shape()[0], (long long)w.attn_qkv_weight->shape()[1]);
    }
    
    // QKV linear: use LoRALinear if initialized, otherwise fallback to base
    TensorPtr qkv;
    if (w.lora_initialized && w.qkv_lin) {
        // Use LoRALinear (includes LoRA delta)
        qkv = w.qkv_lin->forward(x);
    } else {
        // Fallback: base weights only
        qkv = matmul(x, w.attn_qkv_weight);
        qkv = add(qkv, w.attn_qkv_bias);
    }
    
    // 🔍 DEBUG: Layer 11 QKV overall statistics
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        const float* qkv_d = qkv->data<float>();
        double sum = 0.0, sum_sq = 0.0;
        for (int64_t i = 0; i < qkv->numel(); ++i) {
            sum += qkv_d[i];
            sum_sq += qkv_d[i] * qkv_d[i];
        }
        float mean = sum / qkv->numel();
        float std = std::sqrt(sum_sq / qkv->numel() - mean * mean);
        printf("[DEBUG] Layer 11 QKV output (before split): mean=%10.6f, std=%10.6f\n", mean, std);
    }
    
    // Split Q/K/V in order: q, k, v
    const float* qkv_data = qkv->data<float>();
    
    auto q = zeros({B, S, C}, kFloat32, kCPU);
    auto k = zeros({B, S, C}, kFloat32, kCPU);
    auto v = zeros({B, S, C}, kFloat32, kCPU);
    
    float* q_data = q->data<float>();
    float* k_data = k->data<float>();
    float* v_data = v->data<float>();
    
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            const float* qkv_ptr = qkv_data + (b * S + s) * 3 * C;
            float* q_ptr = q_data + (b * S + s) * C;
            float* k_ptr = k_data + (b * S + s) * C;
            float* v_ptr = v_data + (b * S + s) * C;
            
            std::memcpy(q_ptr, qkv_ptr, C * sizeof(float));           // q
            std::memcpy(k_ptr, qkv_ptr + C, C * sizeof(float));       // k
            std::memcpy(v_ptr, qkv_ptr + 2 * C, C * sizeof(float));   // v
        }
    }
    
    // 🔍 DEBUG: Layer 11 QKV statistics after split
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        auto calc_stats = [](const TensorPtr& t, const char* name) {
            const float* data = t->data<float>();
            double sum = 0.0, sum_sq = 0.0;
            for (int64_t i = 0; i < t->numel(); ++i) {
                sum += data[i];
                sum_sq += data[i] * data[i];
            }
            float mean = sum / t->numel();
            float std = std::sqrt(sum_sq / t->numel() - mean * mean);
            printf("[DEBUG] Layer 11 %s (before reshape): mean=%10.6f, std=%10.6f\n", name, mean, std);
        };
        calc_stats(q, "Q");
        calc_stats(k, "K");
        calc_stats(v, "V");
    }
    
    // Rearrange into multi-head: [B,S,C] → [B,S,n_head,Hd] → [B,n_head,S,Hd]
    q = reshape(q, {B, S, n_head, Hd});
    k = reshape(k, {B, S, n_head, Hd});
    v = reshape(v, {B, S, n_head, Hd});
    
    q = permute(q, {0, 2, 1, 3});  // [B,n_head,S,Hd] - permute is already contiguous
    k = permute(k, {0, 2, 1, 3});
    v = permute(v, {0, 2, 1, 3});
    
    // ========== Attention computation: memory-efficient vs standard ==========
    TensorPtr context;
    
    if (config_.use_memory_efficient_attention) {
        // 🚀 Memory-efficient attention: streamed softmax without materializing S×S
        // Memory cost: O(B·H·S·D) vs standard O(B·H·S²)
        MemoryEfficientAttentionConfig attn_config;
        attn_config.use_causal_mask = true;
        attn_config.scale = 1.0f / std::sqrt(static_cast<float>(Hd));
        
        context = memory_efficient_attention(q, k, v, causal_mask, attn_config);
        
        // Skip standard-path debug prints (memory-efficient variant does not produce scores/probs)
    } else {
        // 📊 Standard attention: materialize full matrix (for comparison/debug)
        const float scale = 1.0f / std::sqrt(static_cast<float>(Hd));
        
        auto k_t = transpose(k, 2, 3);  // [B,n_head,Hd,S]
        auto scores = matmul(q, k_t);   // [B,n_head,S,S]
        scores = mul(scores, scale);
    
        // 🔍 DEBUG: Print first head scores 4x4 before applying mask
        if (GPT2_DEBUG_PRINT && block_idx == 0) {
            const float* scores_data = scores->data<float>();
            std::cout << "\n[DEBUG] scores[0,0,0:4,0:4] BEFORE mask:" << std::endl;
            int64_t print_size = std::min(static_cast<int64_t>(4), S);
            for (int64_t i = 0; i < print_size; ++i) {
                for (int64_t j = 0; j < print_size; ++j) {
                    printf("%8.4f ", scores_data[i * S + j]);
                }
                std::cout << std::endl;
            }
        }
        
        // Add masks (fp32, before softmax)
        scores = add(scores, causal_mask);  // Broadcast [S,S] → [B,n_head,S,S]
        if (pad_mask) {
            scores = add(scores, pad_mask);
        }
        
        // Softmax
        auto probs = softmax(scores, -1);  // [B,n_head,S,S]
        
        // context = probs @ v
        context = matmul(probs, v);  // [B,n_head,S,Hd]
    }
    
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        printf("[DEBUG] Layer 11 context shape: [%lld,%lld,%lld,%lld]\n", 
               (long long)context->shape()[0], (long long)context->shape()[1], (long long)context->shape()[2], (long long)context->shape()[3]);
        
        // Print the same position from matmul output
        const float* ctx_data = context->data<float>();
        printf("[DEBUG] Layer 11 matmul output context[0,0,2,0:4]:  ");
        int row = 2;
        for (int d = 0; d < 4; ++d) printf("%.4f ", ctx_data[row * Hd + d]);
        printf("\n");
    }
    
    // 🔍 DEBUG: Layer 11 context statistics (multi-head layout)
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        const float* ctx_data = context->data<float>();
        double sum = 0.0, sum_sq = 0.0;
        for (int64_t i = 0; i < context->numel(); ++i) {
            sum += ctx_data[i];
            sum_sq += ctx_data[i] * ctx_data[i];
        }
        float mean = sum / context->numel();
        float std = std::sqrt(sum_sq / context->numel() - mean * mean);
        printf("[DEBUG] Layer 11 context (multi-head): mean=%10.6f, std=%10.6f\n", mean, std);
    }
    
    // Merge heads: [B,n_head,S,Hd] → [B,S,n_head,Hd] → [B,S,C]
    context = permute(context, {0, 2, 1, 3});  // [B,S,n_head,Hd]
    context = reshape(context, {B, S, C});
    
    // Attention output projection: use LoRALinear when available
    TensorPtr attn_out;
    if (w.lora_initialized && w.proj_lin) {
        attn_out = w.proj_lin->forward(context);  // ✅ Includes LoRA delta
    } else {
        attn_out = matmul(context, w.attn_proj_weight);
        attn_out = add(attn_out, w.attn_proj_bias);
    }
    
    // 🔍 DEBUG: Layer 11 detailed statistics
    if (GPT2_DEBUG_PRINT && block_idx == 11) {
        // Context (after softmax @ v)
        const float* ctx_data = context->data<float>();
        int64_t ctx_n = context->numel();
        double ctx_sum = 0.0;
        for (int64_t j = 0; j < ctx_n; ++j) ctx_sum += ctx_data[j];
        float ctx_mean = ctx_sum / ctx_n;
        printf("[DEBUG] Layer 11 context (before proj): mean=%10.6f\n", ctx_mean);
        
        // Attention projection weight statistics
        const float* proj_w_data = w.attn_proj_weight->data<float>();
        int64_t proj_n = w.attn_proj_weight->numel();
        double proj_sum = 0.0;
        for (int64_t j = 0; j < proj_n; ++j) proj_sum += proj_w_data[j];
        float proj_mean = proj_sum / proj_n;
        printf("[DEBUG] Layer 11 attn_proj_weight: mean=%10.6f, shape=[%lld,%lld]\n", 
               proj_mean, (long long)w.attn_proj_weight->shape()[0], (long long)w.attn_proj_weight->shape()[1]);
        
        // Attention output
        const float* attn_data = attn_out->data<float>();
        int64_t n = attn_out->numel();
        double sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += attn_data[j];
        float mean = sum / n;
        double sum_sq = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double diff = attn_data[j] - mean;
            sum_sq += diff * diff;
        }
        float std = std::sqrt(sum_sq / n);
        printf("[DEBUG] Layer 11 attn output: mean=%10.6f, std=%10.6f\n", mean, std);
    }
    
    // Residual
    auto x_attn = add(residual, attn_out);
    
    // ========== MLP ==========
    residual = x_attn;
    x = layer_norm(x_attn, w.ln_2_weight, w.ln_2_bias);
    
    // MLP fc_in: use LoRALinear when available
    TensorPtr h;
    if (w.lora_initialized && w.fc_in_lin) {
        h = w.fc_in_lin->forward(x);  // ✅ Includes LoRA delta
    } else {
        h = matmul(x, w.mlp_fc_in_weight);
        h = add(h, w.mlp_fc_in_bias);
    }
    
    // GELU new
    h = gelu_new(h);
    
    // MLP fc_out: use LoRALinear when available
    TensorPtr mlp_out;
    if (w.lora_initialized && w.fc_out_lin) {
        mlp_out = w.fc_out_lin->forward(h);  // ✅ Includes LoRA delta
    } else {
        mlp_out = matmul(h, w.mlp_fc_out_weight);
        mlp_out = add(mlp_out, w.mlp_fc_out_bias);
    }
    
    // Residual
    auto x_out = add(residual, mlp_out);
    
    return x_out;
}

// ============================================================================
// Utilities and debugging
// ============================================================================

void GPT2Model::print_model_info() const {
    std::cout << "\n[GPT2Model Info]" << std::endl;
    std::cout << "  vocab_size: " << config_.vocab_size << std::endl;
    std::cout << "  n_positions: " << config_.n_positions << std::endl;
    std::cout << "  n_embd: " << config_.n_embd << std::endl;
    std::cout << "  n_layer: " << config_.n_layer << std::endl;
    std::cout << "  n_head: " << config_.n_head << std::endl;
    std::cout << "  layernorm_eps: " << config_.layernorm_eps << std::endl;
    std::cout << "  tie_word_embeddings: " << (config_.tie_word_embeddings ? "true" : "false") << std::endl;
    std::cout << "  weights_tied: " << (weights_tied_ ? "true" : "false") << std::endl;
    std::cout << "  🚀 use_memory_efficient_attention: " << (config_.use_memory_efficient_attention ? "ON" : "OFF") << std::endl;
    std::cout << "  💾 use_bf16_activations: " << (config_.use_bf16_activations ? "ON" : "OFF") << std::endl;
}

std::vector<TensorPtr> GPT2Model::parameters() {
    std::vector<TensorPtr> params;
    params.push_back(wte_weight_);
    params.push_back(wpe_weight_);
    
    for (auto& block : blocks_) {
        params.push_back(block.ln_1_weight);
        params.push_back(block.ln_1_bias);
        params.push_back(block.attn_qkv_weight);
        params.push_back(block.attn_qkv_bias);
        params.push_back(block.attn_proj_weight);
        params.push_back(block.attn_proj_bias);
        params.push_back(block.ln_2_weight);
        params.push_back(block.ln_2_bias);
        params.push_back(block.mlp_fc_in_weight);
        params.push_back(block.mlp_fc_in_bias);
        params.push_back(block.mlp_fc_out_weight);
        params.push_back(block.mlp_fc_out_bias);
    }
    
    params.push_back(ln_f_weight_);
    params.push_back(ln_f_bias_);
    
    return params;
}

}  // namespace ops
