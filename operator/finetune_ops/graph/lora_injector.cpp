/**
 * @file lora_injector.cpp
 * @brief LoRA injector implementation
 */

#include "lora_injector.h"
#include "../core/ops.h"
#include <iostream>
#include <random>
#include <cmath>

namespace ops {

// ============================================================================
// LoraState initialization
// ============================================================================

void LoraState::init(int64_t in_features, int64_t out_features,
                     int rank, float alpha, float dropout_p) {
    this->scale = alpha / static_cast<float>(rank);
    this->dropout_p = dropout_p;
    this->enabled = true;
    
    // A ~ N(0, 1/r)
    A = std::make_shared<Tensor>(std::vector<int64_t>{in_features, rank}, kFloat32, kCPU);
    float std_a = 1.0f / std::sqrt(static_cast<float>(rank));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_a);
    
    float* a_data = A->data<float>();
    for (int64_t i = 0; i < A->numel(); ++i) {
        a_data[i] = dist(gen);
    }
    
    // B = 0
    B = zeros({rank, out_features}, kFloat32, kCPU);
    
    A->set_requires_grad(true);
    B->set_requires_grad(true);
}

// ============================================================================
// LoraInjector core implementation
// ============================================================================

void LoraInjector::inject(GPT2Model& model, const LoraSpec& spec) {
    spec_ = spec;
    num_layers_ = model.config().n_layer;
    
    const int C = model.config().n_embd;
    const int rank = spec.rank;
    const float scale = spec.alpha / float(rank);  // LoRA scaling factor
    
    // Ensure LoRA modules are initialized
    model.init_lora_modules();
    
    // Decide which layers to inject
    std::vector<int> target_layers;
    if (spec.layers.empty()) {
        for (int i = 0; i < num_layers_; ++i) {
            target_layers.push_back(i);
        }
    } else {
        target_layers = spec.layers;
    }
    
    // LoRA parameter initializer (Kaiming uniform)
    auto create_lora_params = [&](int in_dim, int out_dim) -> std::pair<TensorPtr, TensorPtr> {
        // A: Kaiming init
        auto A = zeros({in_dim, rank}, kFloat32, kCPU);
        float* A_data = A->data<float>();
        float bound = std::sqrt(6.0f / (in_dim + rank));
        std::mt19937 gen(42 + in_dim + out_dim);  // simple seed
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (int64_t i = 0; i < A->numel(); ++i) {
            A_data[i] = dist(gen);
        }
        
        // B: initialize to zeros (no effect)
        auto B = zeros({rank, out_dim}, kFloat32, kCPU);
        
        return {A, B};
    };
    
    int lora_count = 0;
    
    // Attach LoRA for each layer
    for (int i : target_layers) {
        auto& block = model.get_block(i);
        
        for (auto target : spec.targets) {
            if (target == LoraTarget::AttnQKV) {
                if (spec.split_qkv) {
                    // Split into three slices: q/k/v
                    auto [Aq, Bq] = create_lora_params(C, C);
                    auto [Ak, Bk] = create_lora_params(C, C);
                    auto [Av, Bv] = create_lora_params(C, C);
                    block.qkv_lin->attach_lora(Aq, Bq, scale, 0, C);    // q
                    block.qkv_lin->attach_lora(Ak, Bk, scale, C, C);    // k
                    block.qkv_lin->attach_lora(Av, Bv, scale, 2*C, C);  // v
                    lora_count += 3;
                } else {
                    // No split, handle as fused
                    auto [A, B] = create_lora_params(C, 3*C);
                    block.qkv_lin->attach_lora(A, B, scale, 0, 3*C);
                    lora_count++;
                }
            }
            
            if (target == LoraTarget::AttnProj) {
                auto [A, B] = create_lora_params(C, C);
                block.proj_lin->attach_lora(A, B, scale, 0, C);
                lora_count++;
            }
            
            if (target == LoraTarget::MlpFcIn) {
                auto [A, B] = create_lora_params(C, 4*C);
                block.fc_in_lin->attach_lora(A, B, scale, 0, 4*C);
                lora_count++;
            }
            
            if (target == LoraTarget::MlpFcOut) {
                auto [A, B] = create_lora_params(4*C, C);
                block.fc_out_lin->attach_lora(A, B, scale, 0, C);
                lora_count++;
            }
        }
    }
    
    std::cout << "[LoraInjector] Injected " << lora_count << " LoRA modules to " 
              << target_layers.size() << " layers" << std::endl;
    
    // 打印LoRA信息
    int64_t total_params = 0;
    auto all_params = model.get_lora_parameters();
    for (const auto& p : all_params) {
        total_params += p->numel();
    }
    
    std::cout << "[LoRA Info]" << std::endl;
    std::cout << "  Rank: " << rank << std::endl;
    std::cout << "  Alpha: " << spec.alpha << std::endl;
    std::cout << "  Scale: " << scale << std::endl;
    std::cout << "  Total LoRA params: " << total_params << std::endl;
    std::cout << "  Trainable param count: " << all_params.size() << std::endl;
}

void LoraInjector::inject_qkv_split(GPT2Model& model, int layer_idx,
                                   int rank, float alpha, float dropout) {
    const int C = model.config().n_embd;
    auto params = model.attn_qkv_params(layer_idx);
    TensorPtr* W = params.first;
    TensorPtr* B = params.second;
    if (!W || !(*W)) throw std::runtime_error("attn_qkv_weight is null");
    if (!B || !(*B)) throw std::runtime_error("attn_qkv_bias is null");

    // q/k/v 各一组（每组 [C, C]），在列维度上偏移 0, C, 2C
    const char* names[3] = {"q", "k", "v"};
    for (int idx = 0; idx < 3; ++idx) {
        Hook hook;
        hook.name = "blocks." + std::to_string(layer_idx) + ".attn." + names[idx];
        hook.W_ptr = W;
        hook.bias_ptr = B;
        hook.col_offset = static_cast<int64_t>(idx) * C;
        hook.col_size = C;
        hook.state.init(C, C, rank, alpha, dropout);
        hooks_.push_back(std::move(hook));
    }
}

void LoraInjector::inject_qkv_fused(GPT2Model& model, int layer_idx,
                                   int rank, float alpha, float dropout) {
    const int C = model.config().n_embd;
    auto params = model.attn_qkv_params(layer_idx);
    Hook hook;
    hook.name = "blocks." + std::to_string(layer_idx) + ".attn.qkv";
    hook.W_ptr = params.first;
    hook.bias_ptr = params.second;
    hook.col_offset = 0;
    hook.col_size = 3 * static_cast<int64_t>(C);
    hook.state.init(C, 3*C, rank, alpha, dropout);
    hooks_.push_back(std::move(hook));
}

void LoraInjector::inject_layer(GPT2Model& model, int layer_idx,
                               const std::string& layer_name,
                               int64_t in_features, int64_t out_features,
                               int rank, float alpha, float dropout) {
    Hook hook;
    hook.name = "blocks." + std::to_string(layer_idx) + "." + layer_name;
    if (layer_name == "attn.proj") {
        auto p = model.attn_proj_params(layer_idx);
        hook.W_ptr = p.first;
        hook.bias_ptr = p.second;
    } else if (layer_name == "mlp.fc_in") {
        auto p = model.mlp_fc_in_params(layer_idx);
        hook.W_ptr = p.first;
        hook.bias_ptr = p.second;
    } else if (layer_name == "mlp.fc_out") {
        auto p = model.mlp_fc_out_params(layer_idx);
        hook.W_ptr = p.first;
        hook.bias_ptr = p.second;
    } else {
        throw std::runtime_error("Unknown layer_name in inject_layer: " + layer_name);
    }

    hook.col_offset = 0;
    hook.col_size = static_cast<int64_t>(out_features);
    hook.state.init(in_features, out_features, rank, alpha, dropout);
    hooks_.push_back(std::move(hook));
}

// ============================================================================
// Merge / Unmerge
// ============================================================================

void LoraInjector::merge() {
    if (merged_) {
        std::cout << "[LoraInjector] Already merged, skipping" << std::endl;
        return;
    }
    
    for (auto& hook : hooks_) {
        if (!hook.W_ptr || !hook.state.enabled) continue;
        // ΔW = A @ B * scale, add only to specified column range
        auto delta = matmul(hook.state.A, hook.state.B);   // [in, out_s]
        delta = mul(delta, hook.state.scale);

        const auto& wshape = (*hook.W_ptr)->shape();
        int64_t in = wshape[0];
        int64_t out = wshape[1];
        int64_t col_size = hook.col_size < 0 ? out : hook.col_size;
        int64_t col_off = hook.col_offset;
        if (col_off + col_size > out) throw std::runtime_error("LoRA merge: column range out of bounds");

        float* Wd = (*hook.W_ptr)->data<float>();
        const float* Dd = delta->data<float>();
        for (int64_t i = 0; i < in; ++i) {
            for (int64_t j = 0; j < col_size; ++j) {
                Wd[i * out + (col_off + j)] += Dd[i * col_size + j];
            }
        }
    }
    
    merged_ = true;
    std::cout << "[LoraInjector] Merged " << hooks_.size() << " LoRA modules" << std::endl;
}

void LoraInjector::unmerge() {
    if (!merged_) {
        std::cout << "[LoraInjector] Not merged, skipping unmerge" << std::endl;
        return;
    }
    
    for (auto& hook : hooks_) {
        if (!hook.W_ptr || !hook.state.enabled) continue;
        // W = W' - ΔW (same range as merge)
        auto delta = matmul(hook.state.A, hook.state.B);
        delta = mul(delta, hook.state.scale);

        const auto& wshape = (*hook.W_ptr)->shape();
        int64_t in = wshape[0];
        int64_t out = wshape[1];
        int64_t col_size = hook.col_size < 0 ? out : hook.col_size;
        int64_t col_off = hook.col_offset;
        if (col_off + col_size > out) throw std::runtime_error("LoRA unmerge: column range out of bounds");

        float* Wd = (*hook.W_ptr)->data<float>();
        const float* Dd = delta->data<float>();
        for (int64_t i = 0; i < in; ++i) {
            for (int64_t j = 0; j < col_size; ++j) {
                Wd[i * out + (col_off + j)] -= Dd[i * col_size + j];
            }
        }
    }
    
    merged_ = false;
    std::cout << "[LoraInjector] Unmerged " << hooks_.size() << " LoRA modules" << std::endl;
}

// ============================================================================
// Parameter collection
// ============================================================================

std::vector<TensorPtr> LoraInjector::collect_lora_parameters() const {
    std::vector<TensorPtr> params;
    params.reserve(hooks_.size() * 2);
    
    for (const auto& hook : hooks_) {
        if (hook.state.enabled) {
            params.push_back(hook.state.A);
            params.push_back(hook.state.B);
        }
    }
    
    return params;
}

// ============================================================================
// LoRA-augmented linear forward (static helper)
// ============================================================================

TensorPtr LoraInjector::lora_linear_forward(const TensorPtr& x,
                                           const TensorPtr& W,
                                           const TensorPtr& bias,
                                           const LoraState* lora,
                                           bool training) {
    // Base: y = x @ W + b
    TensorPtr y = matmul(x, W);
    if (bias) {
        y = add(y, bias);
    }
    
    // LoRA delta (if present and enabled)
    if (lora && lora->enabled) {
        TensorPtr x_lora = x;
        
        // Dropout (LoRA branch only, active during training)
        if (training && lora->dropout_p > 0.0f) {
            x_lora = dropout(x, lora->dropout_p, training);
        }
        
        // LoRA path: (x @ A) @ B * scale
        auto hidden = matmul(x_lora, lora->A);     // [*, in] @ [in, r] = [*, r]
        auto lora_out = matmul(hidden, lora->B);   // [*, r] @ [r, out] = [*, out]
        lora_out = mul(lora_out, lora->scale);
        
        // Merge with base output
        y = add(y, lora_out);
    }
    
    return y;
}

// ============================================================================
// Debug and info
// ============================================================================

void LoraInjector::print_info() const {
    std::cout << "\n[LoRA Injector Info]" << std::endl;
    std::cout << "  Total hooks: " << hooks_.size() << std::endl;
    std::cout << "  Rank: " << spec_.rank << std::endl;
    std::cout << "  Alpha: " << spec_.alpha << std::endl;
    std::cout << "  Scale: " << (spec_.alpha / spec_.rank) << std::endl;
    std::cout << "  Dropout: " << spec_.dropout << std::endl;
    std::cout << "  Split QKV: " << (spec_.split_qkv ? "true" : "false") << std::endl;
    std::cout << "  Merged: " << (merged_ ? "true" : "false") << std::endl;
    
    int64_t total_lora_params = 0;
    for (const auto& hook : hooks_) {
        if (hook.state.enabled) {
            total_lora_params += hook.state.A->numel() + hook.state.B->numel();
        }
    }
    std::cout << "  Total LoRA params: " << total_lora_params << std::endl;
}

std::vector<TensorPtr> LoraInjector::get_trainable_params() {
    std::vector<TensorPtr> params;
    for (const auto& hook : hooks_) {
        if (hook.state.enabled) {
            params.push_back(hook.state.A);
            params.push_back(hook.state.B);
        }
    }
    return params;
}

// ============================================================================
// Save / Load (TODO: complete safetensors serialization)
// ============================================================================

void LoraInjector::save_lora_safetensors(const std::string& path) const {
    // TODO: implement safetensors writing
    // Key format: lora.blocks.{i}.{target}.{q/k/v}.A, .B
    // Metadata: meta.rank, meta.alpha, meta.dropout, meta.split_qkv
    std::cout << "[LoraInjector] save_lora_safetensors: TODO (path=" << path << ")" << std::endl;
}

void LoraInjector::load_lora_safetensors(const std::string& path) {
    // TODO: load LoRA A/B from safetensors
    std::cout << "[LoraInjector] load_lora_safetensors: TODO (path=" << path << ")" << std::endl;
}

void LoraInjector::merge_all(GPT2Model& model) {
    const auto& cfg = model.config();
    for (int i = 0; i < cfg.n_layer; ++i) {
        auto& blk = model.get_block(i);
        if (blk.qkv_lin) blk.qkv_lin->merge_to_base();
        if (blk.proj_lin) blk.proj_lin->merge_to_base();
        if (blk.fc_in_lin) blk.fc_in_lin->merge_to_base();
        if (blk.fc_out_lin) blk.fc_out_lin->merge_to_base();
    }
}

void LoraInjector::unmerge_all(GPT2Model& model) {
    const auto& cfg = model.config();
    for (int i = 0; i < cfg.n_layer; ++i) {
        auto& blk = model.get_block(i);
        if (blk.qkv_lin) blk.qkv_lin->unmerge_from_base();
        if (blk.proj_lin) blk.proj_lin->unmerge_from_base();
        if (blk.fc_in_lin) blk.fc_in_lin->unmerge_from_base();
        if (blk.fc_out_lin) blk.fc_out_lin->unmerge_from_base();
    }
}

}  // namespace ops

