/**
 * @file qwen_model.cpp
 * @brief Minimal Qwen2.5-0.5B implementation (CPU/BLAS, attention-only LoRA)
 */

#include "qwen_model.h"
#include "../core/checkpoint.h"
#include "../core/autograd_engine.h"
#include "../core/memory_efficient_attention.h"
#include "../core/ops.h"
#include "../core/utils.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <vector>

namespace ops {

// -------------- Config loader (minimal JSON parsing) --------------
QwenConfig QwenConfig::from_pretrained(const std::string& path) {
    QwenConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config: " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string s = buf.str();
    auto get_int = [&](const std::string& key, int default_val) {
        std::regex pat("\"" + key + "\"\\s*:\\s*(\\d+)");
        std::smatch m;
        if (std::regex_search(s, m, pat)) return std::stoi(m[1].str());
        return default_val;
    };
    auto get_float = [&](const std::string& key, float def) {
        std::regex pat("\"" + key + "\"\\s*:\\s*([0-9eE\\.+-]+)");
        std::smatch m;
        if (std::regex_search(s, m, pat)) return std::stof(m[1].str());
        return def;
    };
    auto get_str = [&](const std::string& key, const std::string& def) {
        std::regex pat("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        std::smatch m;
        if (std::regex_search(s, m, pat)) return m[1].str();
        return def;
    };
    cfg.vocab_size = get_int("vocab_size", cfg.vocab_size);
    cfg.hidden_size = get_int("hidden_size", cfg.hidden_size);
    cfg.intermediate_size = get_int("intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = get_int("num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = get_int("num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = get_int("num_key_value_heads", cfg.num_key_value_heads);
    cfg.max_position_embeddings = get_int("max_position_embeddings", cfg.max_position_embeddings);
    cfg.bos_token_id = get_int("bos_token_id", cfg.bos_token_id);
    cfg.eos_token_id = get_int("eos_token_id", cfg.eos_token_id);
    cfg.pad_token_id = get_int("pad_token_id", cfg.pad_token_id);
    cfg.rms_norm_eps = get_float("rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta = get_float("rope_theta", cfg.rope_theta);
    cfg.hidden_act = get_str("hidden_act", cfg.hidden_act);
    return cfg;
}

// -------------- Construction --------------
QwenModel::QwenModel(const QwenConfig& cfg) : config_(cfg) {
    embed_tokens_ = std::make_shared<Tensor>(std::vector<int64_t>{cfg.vocab_size, cfg.hidden_size}, kFloat32, kCPU);
    final_norm_weight_ = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size}, kFloat32, kCPU);
    layers_.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& b = layers_[i];
        b.input_norm_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size}, kFloat32, kCPU);
        b.post_norm_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size}, kFloat32, kCPU);
        int64_t Hd = cfg.hidden_size / cfg.num_attention_heads;
        int64_t kv_out = cfg.num_key_value_heads * Hd;
        b.q_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, cfg.hidden_size}, kFloat32, kCPU);
        b.q_proj_bias = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size}, kFloat32, kCPU);
        b.k_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, kv_out}, kFloat32, kCPU);
        b.k_proj_bias = std::make_shared<Tensor>(std::vector<int64_t>{kv_out}, kFloat32, kCPU);
        b.v_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, kv_out}, kFloat32, kCPU);
        b.v_proj_bias = std::make_shared<Tensor>(std::vector<int64_t>{kv_out}, kFloat32, kCPU);
        b.o_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, cfg.hidden_size}, kFloat32, kCPU);
        b.gate_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, cfg.intermediate_size}, kFloat32, kCPU);
        b.up_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.hidden_size, cfg.intermediate_size}, kFloat32, kCPU);
        b.down_proj_weight = std::make_shared<Tensor>(std::vector<int64_t>{cfg.intermediate_size, cfg.hidden_size}, kFloat32, kCPU);
    }
}

// -------------- Weight dispatch --------------
void QwenModel::assign_weight(const std::string& key, const TensorPtr& tensor) {
    if (key == "embed_tokens.weight") embed_tokens_ = tensor;
    else if (key == "final_norm.weight") final_norm_weight_ = tensor;
    else {
        std::regex pat(R"(layers\.(\d+)\.(.+))");
        std::smatch m;
        if (std::regex_match(key, m, pat)) {
            int idx = std::stoi(m[1].str());
            auto sub = m[2].str();
            if (idx < 0 || idx >= config_.num_hidden_layers) throw std::runtime_error("bad layer idx");
            auto& b = layers_[idx];
            if (sub == "input_norm.weight") b.input_norm_weight = tensor;
            else if (sub == "post_norm.weight") b.post_norm_weight = tensor;
            else if (sub == "self_attn.q_proj.weight") b.q_proj_weight = tensor;
            else if (sub == "self_attn.q_proj.bias") b.q_proj_bias = tensor;
            else if (sub == "self_attn.k_proj.weight") b.k_proj_weight = tensor;
            else if (sub == "self_attn.k_proj.bias") b.k_proj_bias = tensor;
            else if (sub == "self_attn.v_proj.weight") b.v_proj_weight = tensor;
            else if (sub == "self_attn.v_proj.bias") b.v_proj_bias = tensor;
            else if (sub == "self_attn.o_proj.weight") b.o_proj_weight = tensor;
            else if (sub == "mlp.gate_proj.weight") b.gate_proj_weight = tensor;
            else if (sub == "mlp.up_proj.weight") b.up_proj_weight = tensor;
            else if (sub == "mlp.down_proj.weight") b.down_proj_weight = tensor;
        }
    }
}

// -------------- LoRA --------------
void QwenModel::init_lora(int rank, float alpha, float dropout, bool qv_only) {
    float scale = alpha / rank;
    for (auto& b : layers_) {
        if (b.lora_initialized) continue;
        
        // Create LoRALinear wrappers
        b.q_lin = std::make_unique<LoRALinear>(&b.q_proj_weight, &b.q_proj_bias);
        b.v_lin = std::make_unique<LoRALinear>(&b.v_proj_weight, &b.v_proj_bias);
        
        if (!qv_only) {
            b.k_lin = std::make_unique<LoRALinear>(&b.k_proj_weight, &b.k_proj_bias);
            b.o_lin = std::make_unique<LoRALinear>(&b.o_proj_weight, nullptr);
        }
        
        auto make_AB = [&](int in_dim, int out_dim) -> std::pair<TensorPtr, TensorPtr> {
            auto A = std::make_shared<Tensor>(std::vector<int64_t>{in_dim, rank}, kFloat32, kCPU);
            auto B = std::make_shared<Tensor>(std::vector<int64_t>{rank, out_dim}, kFloat32, kCPU);
            A->set_requires_grad(true); B->set_requires_grad(true);
            // init A ~ U(-0.01,0.01), B=0
            auto tmpA = uniform({in_dim, rank}, -0.01f, 0.01f, kFloat32, kCPU);
            std::memcpy(A->data<void>(), tmpA->data<void>(), sizeof(float) * in_dim * rank);
            std::vector<float> zeros(rank * out_dim, 0.0f);
            std::memcpy(B->data<void>(), zeros.data(), sizeof(float) * rank * out_dim);
            return {A, B};
        };
        
        int64_t Hd = config_.hidden_size / config_.num_attention_heads;
        int64_t kv_out = config_.num_key_value_heads * Hd;
        
        // Q and V always carry LoRA
        auto [Aq, Bq] = make_AB(config_.hidden_size, config_.hidden_size);
        auto [Av, Bv] = make_AB(config_.hidden_size, kv_out);
        b.q_lin->attach_lora(Aq, Bq, scale);
        b.v_lin->attach_lora(Av, Bv, scale);
        
        // K and O only carry LoRA when qv_only is false
        if (!qv_only) {
            auto [Ak, Bk] = make_AB(config_.hidden_size, kv_out);
            auto [Ao, Bo] = make_AB(config_.hidden_size, config_.hidden_size);
            b.k_lin->attach_lora(Ak, Bk, scale);
            b.o_lin->attach_lora(Ao, Bo, scale);
        }
        
        b.lora_initialized = true;
    }
    
    std::cout << "[LoRA] Initialized with rank=" << rank << ", alpha=" << alpha 
              << ", qv_only=" << (qv_only ? "true" : "false") << std::endl;
}

std::vector<TensorPtr> QwenModel::get_lora_parameters() const {
    std::vector<TensorPtr> params;
    for (const auto& b : layers_) {
        if (!b.lora_initialized) continue;
        auto add = [&](const std::unique_ptr<LoRALinear>& lin) {
            if (!lin) return;
            // Only layers with attached LoRA have trainable parameters
            if (lin->slices().empty()) return;
            auto ps = lin->trainable_parameters();
            params.insert(params.end(), ps.begin(), ps.end());
        };
        // Q and V always carry LoRA
        add(b.q_lin); 
        add(b.v_lin);
        // K and O may be absent when qv_only mode is enabled
        if (b.k_lin) add(b.k_lin); 
        if (b.o_lin) add(b.o_lin);
    }
    return params;
}

void QwenModel::freeze_base() {
    auto freeze = [](const TensorPtr& t) {
        if (t) t->set_requires_grad(false);
    };
    freeze(embed_tokens_);
    freeze(final_norm_weight_);
    for (auto& b : layers_) {
        freeze(b.input_norm_weight);
        freeze(b.post_norm_weight);
        freeze(b.q_proj_weight); freeze(b.q_proj_bias);
        freeze(b.k_proj_weight); freeze(b.k_proj_bias);
        freeze(b.v_proj_weight); freeze(b.v_proj_bias);
        freeze(b.o_proj_weight);
        freeze(b.gate_proj_weight);
        freeze(b.up_proj_weight);
        freeze(b.down_proj_weight);
    }
}

// -------------- Helper functions --------------
TensorPtr QwenModel::embedding_lookup(const TensorPtr& weight, const TensorPtr& indices) {
    // [B,S] int32 -> [B,S,H]
    auto shape = indices->shape();
    int64_t B = shape[0], S = shape[1], H = weight->shape()[1];
    auto out = zeros({B, S, H}, kFloat32, kCPU);
    const int32_t* idx = indices->data<int32_t>();
    const float* w = weight->data<float>();
    float* o = out->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = idx[b * S + s];
            const float* src = w + id * H;
            std::memcpy(o + (b * S + s) * H, src, sizeof(float) * H);
        }
    }
    return out;
}

TensorPtr QwenModel::build_causal_mask(int seq_len) {
    return ops::create_causal_mask(seq_len, kFloat32, kCPU);
}

TensorPtr QwenModel::attention(const TensorPtr& x, QwenBlock& blk,
                               const TensorPtr& causal_mask,
                               const TensorPtr& pad_mask, int64_t seq_len) {
    int64_t B = x->shape()[0];
    int64_t S = x->shape()[1];
    int64_t C = x->shape()[2];
    int64_t H = config_.num_attention_heads;
    int64_t HKV = config_.num_key_value_heads;
    int64_t Hd = C / H;
    // int64_t kv_out = HKV * Hd;

    const bool debug = (std::getenv("QWEN_DEBUG") != nullptr);
    auto tensor_stats = [](const TensorPtr& t, const char* name) {
        const float* p = t->data<float>();
        double sum = 0.0, sumsq = 0.0;
        int64_t n = t->numel();
        for (int64_t i = 0; i < n; ++i) {
            sum += p[i];
            sumsq += p[i] * p[i];
        }
        double mean = sum / static_cast<double>(n);
        double std = std::sqrt(sumsq / static_cast<double>(n) - mean * mean);
        std::cout << name << " mean " << mean << " std " << std << " first8=[";
        int64_t take = std::min<int64_t>(8, n);
        for (int64_t i = 0; i < take; ++i) {
            if (i) std::cout << ",";
            std::cout << p[i];
        }
        std::cout << "]" << std::endl;
    };
    auto tensor_slice_4d = [](const TensorPtr& t, const char* name, int64_t b, int64_t h, int64_t s, int64_t take = 8) {
        const auto& shape = t->shape();
        if (shape.size() != 4) return;
        if (b >= shape[0] || h >= shape[1] || s >= shape[2]) return;
        const float* p = t->data<float>();
        int64_t width = shape[3];
        int64_t base = ((b * shape[1] + h) * shape[2] + s) * width;
        std::cout << name << "[" << b << "," << h << "," << s << ",:8]=[";
        int64_t n = std::min<int64_t>(take, width);
        for (int64_t i = 0; i < n; ++i) {
            if (i) std::cout << ",";
            std::cout << p[base + i];
        }
        std::cout << "]" << std::endl;
    };
    auto tensor_slice_3d = [](const TensorPtr& t, const char* name, int64_t b, int64_t s, int64_t take = 8) {
        const auto& shape = t->shape();
        if (shape.size() != 3) return;
        if (b >= shape[0] || s >= shape[1]) return;
        const float* p = t->data<float>();
        int64_t width = shape[2];
        int64_t base = (b * shape[1] + s) * width;
        std::cout << name << "[" << b << "," << s << ",:8]=[";
        int64_t n = std::min<int64_t>(take, width);
        for (int64_t i = 0; i < n; ++i) {
            if (i) std::cout << ",";
            std::cout << p[base + i];
        }
        std::cout << "]" << std::endl;
    };

    auto q = blk.lora_initialized && blk.q_lin ? blk.q_lin->forward(x) : add(matmul(x, blk.q_proj_weight), blk.q_proj_bias);
    auto k = blk.lora_initialized && blk.k_lin ? blk.k_lin->forward(x) : add(matmul(x, blk.k_proj_weight), blk.k_proj_bias);
    auto v = blk.lora_initialized && blk.v_lin ? blk.v_lin->forward(x) : add(matmul(x, blk.v_proj_weight), blk.v_proj_bias);

    static bool attn_debug_done = false;
    if (debug && !attn_debug_done) {
        tensor_stats(q, "q_proj");
        tensor_stats(k, "k_proj");
        tensor_stats(v, "v_proj");
    }

    q = reshape(q, {B, S, H, Hd});
    q = permute(q, {0, 2, 1, 3});  // [B,H,S,Hd]

    k = reshape(k, {B, S, HKV, Hd});
    v = reshape(v, {B, S, HKV, Hd});
    k = permute(k, {0, 2, 1, 3});  // [B,HKV,S,Hd]
    v = permute(v, {0, 2, 1, 3});

    // GQA: repeat kv heads to match H
    if (HKV != H) {
        int repeat = H / HKV;
        k = repeat_kv_heads(k, repeat); // [B,H,S,Hd]
        v = repeat_kv_heads(v, repeat);
    }

    // Qwen2/Qwen2.5 uses rotate_half-style RoPE: pair front-half dims with back-half dims.
    q = apply_rope_contiguous(q, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta);
    k = apply_rope_contiguous(k, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta);
    if (debug && !attn_debug_done) {
        tensor_stats(q, "q_rope");
        tensor_stats(k, "k_rope");
        tensor_slice_4d(q, "q_rope", 0, 0, 1);
        tensor_slice_4d(q, "q_rope", 0, 0, 2);
        tensor_slice_4d(k, "k_rope", 0, 0, 1);
    }

    TensorPtr ctx;
    if (efficient_attention_) {
        MemoryEfficientAttentionConfig options;
        options.scale = 1.0f / std::sqrt(static_cast<float>(Hd));
        ctx = memory_efficient_attention(q, k, v, causal_mask, pad_mask, options);
        ++efficient_attention_calls_;
    } else {
    auto k_t = transpose(k, 2, 3); // [B,H,Hd,S]
    auto scores = matmul(q, k_t);  // [B,H,S,S]
    scores = mul(scores, 1.0f / std::sqrt(static_cast<float>(Hd)));

    if (causal_mask) scores = add(scores, causal_mask);
    if (pad_mask) scores = add(scores, pad_mask);
    auto probs = softmax(scores, -1);
    if (debug && !attn_debug_done) {
        tensor_stats(scores, "attn_scores");
        tensor_stats(probs, "attn_probs");
        tensor_slice_4d(scores, "attn_scores", 0, 0, 1);
        tensor_slice_4d(scores, "attn_scores", 0, 0, 2);
        tensor_slice_4d(probs, "attn_probs", 0, 0, 1);
        tensor_slice_4d(probs, "attn_probs", 0, 0, 2);
    }
    ctx = matmul(probs, v);  // [B,H,S,Hd]
    }
    ctx = permute(ctx, {0, 2, 1, 3});
    ctx = reshape(ctx, {B, S, C});

    auto out = blk.lora_initialized && blk.o_lin ? blk.o_lin->forward(ctx) : matmul(ctx, blk.o_proj_weight);
    if (debug && !attn_debug_done) {
        tensor_stats(ctx, "attn_ctx");
        tensor_stats(out, "attn_o_proj");
        tensor_slice_3d(ctx, "attn_ctx", 0, 1);
        tensor_slice_3d(out, "attn_o_proj", 0, 1);
        attn_debug_done = true;
    }
    return out;
}

TensorPtr QwenModel::mlp(const TensorPtr& x, QwenBlock& blk) {
    const bool debug = (std::getenv("QWEN_DEBUG") != nullptr);
    auto tensor_stats = [](const TensorPtr& t, const char* name) {
        const float* p = t->data<float>();
        double sum = 0.0, sumsq = 0.0;
        int64_t n = t->numel();
        for (int64_t i = 0; i < n; ++i) {
            sum += p[i];
            sumsq += p[i] * p[i];
        }
        double mean = sum / static_cast<double>(n);
        double std = std::sqrt(sumsq / static_cast<double>(n) - mean * mean);
        std::cout << name << " mean " << mean << " std " << std << " first8=[";
        int64_t take = std::min<int64_t>(8, n);
        for (int64_t i = 0; i < take; ++i) {
            if (i) std::cout << ",";
            std::cout << p[i];
        }
        std::cout << "]" << std::endl;
    };
    auto gate = matmul(x, blk.gate_proj_weight);
    auto up = matmul(x, blk.up_proj_weight);
    auto act = swiglu(gate, up);
    auto down = matmul(act, blk.down_proj_weight);
    static bool mlp_debug_done = false;
    if (debug && !mlp_debug_done) {
        tensor_stats(gate, "gate_proj");
        tensor_stats(up, "up_proj");
        tensor_stats(act, "swiglu");
        tensor_stats(down, "down_proj");
        mlp_debug_done = true;
    }
    return down;
}

std::vector<std::pair<std::string, TensorPtr*>> QwenModel::block_parameters(int index) {
    auto& b = layers_.at(static_cast<size_t>(index));
    const auto prefix = "layers." + std::to_string(index) + ".";
    return {
        {prefix + "input_norm.weight", &b.input_norm_weight},
        {prefix + "post_norm.weight", &b.post_norm_weight},
        {prefix + "self_attn.q_proj.weight", &b.q_proj_weight},
        {prefix + "self_attn.q_proj.bias", &b.q_proj_bias},
        {prefix + "self_attn.k_proj.weight", &b.k_proj_weight},
        {prefix + "self_attn.k_proj.bias", &b.k_proj_bias},
        {prefix + "self_attn.v_proj.weight", &b.v_proj_weight},
        {prefix + "self_attn.v_proj.bias", &b.v_proj_bias},
        {prefix + "self_attn.o_proj.weight", &b.o_proj_weight},
        {prefix + "mlp.gate_proj.weight", &b.gate_proj_weight},
        {prefix + "mlp.up_proj.weight", &b.up_proj_weight},
        {prefix + "mlp.down_proj.weight", &b.down_proj_weight},
    };
}

void QwenModel::enable_parameter_sharding(const sharding::ShardConfig& config) {
    if (sharder_) throw std::runtime_error("Qwen parameter sharding is already enabled");
    auto bytes = [](const TensorPtr& t) -> size_t {
        if (!t || t->dtype() != kFloat32 || t->requires_grad()) {
            throw std::runtime_error("Qwen sharding requires loaded, frozen FP32 base tensors");
        }
        return static_cast<size_t>(t->numel()) * sizeof(float);
    };
    // Preload each complete block; a smaller budget could evict an owner before use.
    size_t working_set = std::max(bytes(embed_tokens_), bytes(final_norm_weight_));
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        size_t block_bytes = 0;
        for (const auto& p : block_parameters(i)) block_bytes += bytes(*p.second);
        working_set = std::max(working_set, block_bytes);
    }
    if (config.max_resident_bytes < working_set) {
        throw std::runtime_error("Qwen shard budget must fit embedding and one complete block; minimum bytes=" + std::to_string(working_set));
    }
    sharder_ = std::make_unique<sharding::ParameterSharder>(config);
    sharder_->register_parameter("embed_tokens.weight", embed_tokens_, false, &embed_tokens_);
    sharder_->register_parameter("final_norm.weight", final_norm_weight_, false, &final_norm_weight_);
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        for (const auto& p : block_parameters(i)) {
            sharder_->register_parameter(p.first, *p.second, false, p.second);
        }
    }
}

void QwenModel::require_block_parameters(int index) {
    if (!sharder_) return;
    for (const auto& p : block_parameters(index)) sharder_->require(p.first);
}

void QwenModel::offload_base_parameters() {
    if (sharder_) sharder_->offload_all();
}

sharding::ShardStats QwenModel::sharding_stats() const {
    return sharder_ ? sharder_->stats() : sharding::ShardStats{};
}

// -------------- Forward --------------
TensorPtr QwenModel::forward_block(const TensorPtr& x, int index, const TensorPtr& causal_mask,
                                 const TensorPtr& pad_mask) {
    ++block_forward_calls_;
    require_block_parameters(index);
    auto& blk = layers_[index];
    auto normed = rms_norm_affine(x, blk.input_norm_weight, config_.rms_norm_eps);
    auto hidden = add(x, attention(normed, blk, causal_mask, pad_mask, x->shape()[1]));
    auto normed2 = rms_norm_affine(hidden, blk.post_norm_weight, config_.rms_norm_eps);
    return add(hidden, mlp(normed2, blk));
}

TensorPtr QwenModel::forward(const TensorPtr& input_ids, const TensorPtr& attention_mask) {
    if (sharder_) sharder_->require("embed_tokens.weight");
    auto x = embedding_lookup(embed_tokens_, input_ids); // [B,S,H]
    const bool checkpoint_active = checkpointing_ && autograd::Engine::instance().is_enabled();
    // Reentrant checkpoint needs a grad-requiring input even with a frozen embedding.
    if (checkpoint_active) x->set_requires_grad(true);

    TensorPtr pad_mask = nullptr;
    if (attention_mask) {
        // attention_mask: [B,S] float32 (1=keep, 0=pad) -> broadcast to [B,1,1,S]
        auto am = attention_mask;
        auto mask = zeros({am->shape()[0], 1, 1, am->shape()[1]}, kFloat32, kCPU);
        const float* src = am->data<float>();
        float* dst = mask->data<float>();
        int64_t B = am->shape()[0], S = am->shape()[1];
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S; ++s) {
                float keep = src[b * S + s];
                float v = (keep > 0.5f) ? 0.0f : -1e9f;
                dst[b * S + s] = v;
            }
        }
        pad_mask = mask;
    }

    const bool debug = (std::getenv("QWEN_DEBUG") != nullptr);
    auto tensor_stats = [](const TensorPtr& t, const char* name) {
        const float* p = t->data<float>();
        double sum=0.0, sumsq=0.0;
        int64_t n=t->numel();
        for(int64_t i=0;i<n;++i){sum+=p[i]; sumsq+=p[i]*p[i];}
        double mean=sum/n; double std=std::sqrt(sumsq/n - mean*mean);
        std::cout << name << " mean " << mean << " std " << std << " first8=[";
        int64_t take = std::min<int64_t>(8, n);
        for (int64_t i = 0; i < take; ++i) {
            if (i) std::cout << ",";
            std::cout << p[i];
        }
        std::cout << "]" << std::endl;
    };

    auto causal_mask = build_causal_mask(x->shape()[1]);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        if (!debug) {
            if (checkpoint_active) {
                x = checkpoint([this, i, causal_mask, pad_mask](const TensorPtr& input) {
                    return forward_block(input, i, causal_mask, pad_mask);
                }, x);
            } else {
                x = forward_block(x, i, causal_mask, pad_mask);
            }
            continue;
        }
        if (checkpoint_active) throw std::runtime_error("QWEN_DEBUG cannot bypass activation checkpointing");
        ++block_forward_calls_;
        require_block_parameters(i);
        auto& blk = layers_[i];
        auto normed = rms_norm_affine(x, blk.input_norm_weight, config_.rms_norm_eps);
        auto attn_out = attention(normed, blk, causal_mask, pad_mask, x->shape()[1]);
        x = add(x, attn_out);
        auto normed2 = rms_norm_affine(x, blk.post_norm_weight, config_.rms_norm_eps);
        auto mlp_out = mlp(normed2, blk);
        x = add(x, mlp_out);

        if (debug && i==0) {
            tensor_stats(normed, "ln1");
            tensor_stats(attn_out, "attn_out");
            tensor_stats(normed2, "ln2");
            tensor_stats(mlp_out, "mlp_out");
        }
    }
    if (sharder_) sharder_->require("final_norm.weight");
    x = rms_norm_affine(x, final_norm_weight_, config_.rms_norm_eps);
    if (sharder_) sharder_->require("embed_tokens.weight");
    // Keep the tied vocabulary weight in [vocab, hidden] layout. The
    // transpose-aware kernel avoids a full weight copy in forward and backward.
    auto logits = matmul_rhs_T(x, embed_tokens_);
    return logits;
}

} // namespace ops
