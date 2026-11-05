/**
 * @file qwen_model.cpp
 * @brief Minimal Qwen2.5-0.5B implementation (CPU/BLAS, attention-only LoRA)
 */

#include "qwen_model.h"
#include "../core/ops.h"
#include "../core/utils.h"
#include "../core/memory_efficient_attention.h"
#include "../core/checkpoint.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <random>
#include <sstream>
#include <vector>

namespace ops {

namespace {

float qwen_weight_value(const TensorPtr& weight, int64_t idx) {
    if (weight->dtype() == kFloat32) {
        return weight->data<float>()[idx];
    }
    if (weight->dtype() == kFloat16) {
        return fp16_bits_to_float32(weight->data<uint16_t>()[idx]);
    }
    if (weight->dtype() == kBFloat16) {
        return bf16_bits_to_float32(weight->data<uint16_t>()[idx]);
    }
    throw std::runtime_error("QwenModel: unsupported weight dtype " + DTypeUtils::to_string(weight->dtype()));
}

} // namespace

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
    if (offload_) throw std::logic_error("load weights before enabling parameter offload");
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
void QwenModel::init_lora(int rank, float alpha, float dropout [[maybe_unused]], bool qv_only, uint64_t seed) {
    if (offload_) throw std::logic_error("initialize LoRA before enabling parameter offload");
    float scale = alpha / rank;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> init_dist(-0.01f, 0.01f);
    for (auto& b : layers_) {
        if (b.lora_initialized) continue;
        
        // Create LoRALinear wrappers
        b.q_lin = std::make_unique<LoRALinear>(b.q_proj_weight, b.q_proj_bias);
        b.v_lin = std::make_unique<LoRALinear>(b.v_proj_weight, b.v_proj_bias);
        
        if (!qv_only) {
            b.k_lin = std::make_unique<LoRALinear>(b.k_proj_weight, b.k_proj_bias);
            b.o_lin = std::make_unique<LoRALinear>(b.o_proj_weight, nullptr);
        }
        
        auto make_AB = [&](int in_dim, int out_dim) -> std::pair<TensorPtr, TensorPtr> {
            auto A = std::make_shared<Tensor>(std::vector<int64_t>{in_dim, rank}, kFloat32, kCPU);
            auto B = std::make_shared<Tensor>(std::vector<int64_t>{rank, out_dim}, kFloat32, kCPU);
            A->set_requires_grad(true); B->set_requires_grad(true);
            // init A ~ U(-0.01,0.01), B=0
            float* a_data = A->data<float>();
            for (int64_t i = 0; i < in_dim * rank; ++i) {
                a_data[i] = init_dist(rng);
            }
            std::memset(B->data<void>(), 0, sizeof(float) * rank * out_dim);
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

std::vector<TensorPtr> QwenModel::parameters() const {
    if (offload_) throw std::logic_error("use get_lora_parameters after enabling frozen base offload");
    std::vector<TensorPtr> params;
    params.reserve(2 + static_cast<size_t>(config_.num_hidden_layers) * 12);
    params.push_back(embed_tokens_);
    params.push_back(final_norm_weight_);
    for (const auto& b : layers_) {
        params.push_back(b.input_norm_weight);
        params.push_back(b.post_norm_weight);
        params.push_back(b.q_proj_weight);
        params.push_back(b.q_proj_bias);
        params.push_back(b.k_proj_weight);
        params.push_back(b.k_proj_bias);
        params.push_back(b.v_proj_weight);
        params.push_back(b.v_proj_bias);
        params.push_back(b.o_proj_weight);
        params.push_back(b.gate_proj_weight);
        params.push_back(b.up_proj_weight);
        params.push_back(b.down_proj_weight);
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
    float* o = out->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = idx[b * S + s];
            float* dst = o + (b * S + s) * H;
            if (weight->dtype() == kFloat32) {
                const float* src = weight->data<float>() + id * H;
                std::memcpy(dst, src, sizeof(float) * H);
            } else {
                const int64_t base = static_cast<int64_t>(id) * H;
                for (int64_t h = 0; h < H; ++h) {
                    dst[h] = qwen_weight_value(weight, base + h);
                }
            }
        }
    }
    return out;
}

TensorPtr QwenModel::build_causal_mask(int seq_len) {
    return ops::create_causal_mask(seq_len, kFloat32, kCPU);
}

TensorPtr QwenModel::attention(const TensorPtr& x, QwenBlock& blk,
                               const TensorPtr& causal_mask,
                               const TensorPtr& pad_mask, int64_t seq_len [[maybe_unused]]) {
    int64_t B = x->shape()[0];
    int64_t S = x->shape()[1];
    int64_t C = x->shape()[2];
    int64_t H = config_.num_attention_heads;
    int64_t HKV = config_.num_key_value_heads;
    int64_t Hd = C / H;
    // int64_t kv_out = HKV * Hd;

    auto q = blk.lora_initialized && blk.q_lin ? blk.q_lin->forward(x) : add(matmul(x, blk.q_proj_weight), blk.q_proj_bias);
    auto k = blk.lora_initialized && blk.k_lin ? blk.k_lin->forward(x) : add(matmul(x, blk.k_proj_weight), blk.k_proj_bias);
    auto v = blk.lora_initialized && blk.v_lin ? blk.v_lin->forward(x) : add(matmul(x, blk.v_proj_weight), blk.v_proj_bias);

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

    q = apply_rope(q, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta);
    k = apply_rope(k, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta);

    TensorPtr ctx;
    if (memory_efficient_attention_) {
        MemoryEfficientAttentionConfig cfg;
        cfg.use_causal_mask = false;
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(Hd));
        ctx = memory_efficient_attention(q, k, v, causal_mask, pad_mask, cfg);
    } else {
    auto k_t = transpose(k, 2, 3); // [B,H,Hd,S]
    auto scores = matmul(q, k_t);  // [B,H,S,S]
    scores = mul(scores, 1.0f / std::sqrt(static_cast<float>(Hd)));

    if (causal_mask) scores = add(scores, causal_mask);
    if (pad_mask) scores = add(scores, pad_mask);
    auto probs = softmax(scores, -1);
    ctx = matmul(probs, v);  // [B,H,S,Hd]
    }
    ctx = permute(ctx, {0, 2, 1, 3});
    ctx = reshape(ctx, {B, S, C});

    auto out = blk.lora_initialized && blk.o_lin ? blk.o_lin->forward(ctx) : matmul(ctx, blk.o_proj_weight);
    return out;
}

TensorPtr QwenModel::mlp(const TensorPtr& x, QwenBlock& blk) {
    auto gate = matmul(x, blk.gate_proj_weight);
    auto up = matmul(x, blk.up_proj_weight);
    auto act = swiglu(gate, up);
    auto down = matmul(act, blk.down_proj_weight);
    return down;
}

// -------------- 前向 --------------
TensorPtr QwenModel::forward_hidden(const TensorPtr& input_ids, const TensorPtr& attention_mask) {
    if (offload_) offload_->load(0);
    auto x = embedding_lookup(embed_tokens_, input_ids); // [B,S,H]
    if (offload_) offload_->offload_all();

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
        std::cout << name << " mean " << mean << " std " << std << std::endl;
    };

    auto causal_mask = build_causal_mask(x->shape()[1]);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        auto block_forward = [this,i,causal_mask,pad_mask,debug,tensor_stats](const TensorPtr& input) {
        if (offload_) offload_->load(static_cast<size_t>(i)+2);
        auto& blk = layers_[i];
        OffloadLinearLease lease{bool(offload_)};
        lease.bind(blk.q_lin.get(),blk.q_proj_weight,blk.q_proj_bias);
        lease.bind(blk.k_lin.get(),blk.k_proj_weight,blk.k_proj_bias);
        lease.bind(blk.v_lin.get(),blk.v_proj_weight,blk.v_proj_bias);
        lease.bind(blk.o_lin.get(),blk.o_proj_weight);
        auto x = input;
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
        return x;
        };
        std::vector<TensorPtr> captures;
        for (auto* lin : {layers_[i].q_lin.get(),layers_[i].k_lin.get(),layers_[i].v_lin.get(),layers_[i].o_lin.get()})
            if(lin) { auto p=lin->trainable_parameters();captures.insert(captures.end(),p.begin(),p.end()); }
        x = activation_checkpointing_ ? checkpoint(block_forward,x,captures) : block_forward(x);
        if (offload_) offload_->offload_all();
    }
    if (offload_) offload_->load(1);
    x = rms_norm_affine(x, final_norm_weight_, config_.rms_norm_eps);
    if (offload_) offload_->offload_all();
    return x;
}

TensorPtr QwenModel::lm_head(const TensorPtr& hidden) {
    if (offload_) offload_->load(0);
    auto output = matmul_rhs_T(hidden, embed_tokens_);
    if (offload_) offload_->offload_all();
    return output;
}

TensorPtr QwenModel::forward(const TensorPtr& input_ids, const TensorPtr& attention_mask) {
    return lm_head(forward_hidden(input_ids, attention_mask));
}

void QwenModel::set_activation_checkpointing(bool enabled) {
#ifndef USE_NEW_AUTOGRAD_ENGINE
    if (enabled) throw std::logic_error("activation checkpointing requires the new autograd engine");
#endif
    activation_checkpointing_ = enabled;
}

void QwenModel::enable_parameter_sharding(const sharding::ShardConfig& config) {
    if (offload_) throw std::logic_error("parameter offload already enabled");
    std::vector<FrozenModelOffload::Group> groups{{&embed_tokens_},{&final_norm_weight_}};
    for(auto& b:layers_) {
        if(!b.lora_initialized) throw std::logic_error("initialize LoRA before offload");
        for(auto* lin:{b.q_lin.get(),b.k_lin.get(),b.v_lin.get(),b.o_lin.get()})
            if(lin && lin->is_merged()) throw std::logic_error("unmerge LoRA before offload");
        groups.push_back({&b.input_norm_weight,&b.post_norm_weight,&b.q_proj_weight,&b.q_proj_bias,
            &b.k_proj_weight,&b.k_proj_bias,&b.v_proj_weight,&b.v_proj_bias,&b.o_proj_weight,
            &b.gate_proj_weight,&b.up_proj_weight,&b.down_proj_weight});
    }
    offload_=std::make_unique<FrozenModelOffload>(config,groups);
    for(auto& b:layers_) for(auto* lin:{b.q_lin.get(),b.k_lin.get(),b.v_lin.get(),b.o_lin.get()})
        if(lin) lin->set_base_weights(nullptr);
    offload_->offload_all();
}

} // namespace ops
