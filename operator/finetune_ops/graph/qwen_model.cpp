/**
 * @file qwen_model.cpp
 * @brief Configuration-driven Qwen2/Qwen3/Qwen3.5 text decoder (CPU/BLAS).
 */

#include "qwen_model.h"
#include "../../third_party/nlohmann/json.hpp"
#include "../core/checkpoint.h"
#include "../core/lm_loss.h"
#include "../core/memory_efficient_attention.h"
#include "../core/ops.h"
#include "../core/qwen35_ops.h"
#include "../core/utils.h"
#include "safetensors_loader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_set>
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
    throw std::runtime_error("QwenModel: unsupported weight dtype " +
                             DTypeUtils::to_string(weight->dtype()));
}

} // namespace

// Configuration describes the architecture, not an individual checkpoint size.
int QwenConfig::attention_head_dim() const {
    if (head_dim > 0)
        return head_dim;
    if (model_type == "qwen3")
        return 128;
    if (hybrid())
        return 256;
    if (num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
        throw std::invalid_argument("Qwen requires head_dim when hidden_size is "
                                    "not divisible by attention heads");
    return hidden_size / num_attention_heads;
}

std::vector<std::string> QwenConfig::default_lora_targets() const {
    validate();
    if (!hybrid())
        return {"q_proj", "k_proj", "v_proj", "o_proj"};
    std::vector<std::string> targets;
    for (const auto& type : layer_types) {
        const std::vector<std::string> names =
            type == "linear_attention"
                ? std::vector<std::string>{"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                                           "out_proj"}
                : std::vector<std::string>{"q_proj", "k_proj", "v_proj", "o_proj"};
        for (const auto& name : names)
            if (std::find(targets.begin(), targets.end(), name) == targets.end())
                targets.push_back(name);
    }
    return targets;
}

void QwenConfig::validate() const {
    if (model_type != "qwen2" && model_type != "qwen3" && !hybrid())
        throw std::invalid_argument("Unsupported Qwen model_type: " + model_type +
                                    "; supported text architectures: qwen2, qwen3, qwen3_5_text");
    if (model_type == "qwen2" && !attention_bias)
        throw std::invalid_argument("Qwen2 requires Q/K/V attention biases");
    if (vocab_size <= 0 || hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 ||
        num_attention_heads <= 0 || num_key_value_heads <= 0 || max_position_embeddings <= 0 ||
        head_dim < 0)
        throw std::invalid_argument("Qwen config dimensions must be positive");
    const int dim = attention_head_dim();
    if (dim <= 0 || dim % 2 || num_attention_heads % num_key_value_heads ||
        int64_t(num_attention_heads) * dim > INT32_MAX ||
        int64_t(num_key_value_heads) * dim > INT32_MAX)
        throw std::invalid_argument("Qwen requires even head_dim and attention "
                                    "heads divisible by KV heads");
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0 || !std::isfinite(rope_theta) ||
        rope_theta <= 0)
        throw std::invalid_argument("Qwen normalization epsilon and RoPE theta "
                                    "must be finite and positive");
    if (hybrid()) {
        if (attention_bias || layer_types.size() != size_t(num_hidden_layers))
            throw std::invalid_argument("Qwen3.5 requires bias-free projections and "
                                        "a complete layer_types list");
        for (const auto& type : layer_types)
            if (type != "full_attention" && type != "linear_attention")
                throw std::invalid_argument("Unknown Qwen3.5 layer type");
        if (linear_num_key_heads <= 0 || linear_num_value_heads <= 0 || linear_key_head_dim <= 0 ||
            linear_value_head_dim <= 0 || linear_conv_kernel_dim <= 0 ||
            linear_num_value_heads % linear_num_key_heads ||
            int64_t(linear_num_key_heads) * linear_key_head_dim * 2 +
                    int64_t(linear_num_value_heads) * linear_value_head_dim >
                INT32_MAX ||
            !std::isfinite(partial_rotary_factor) || partial_rotary_factor <= 0 ||
            partial_rotary_factor > 1 || int(dim * partial_rotary_factor) <= 0 ||
            int(dim * partial_rotary_factor) % 2)
            throw std::invalid_argument("Invalid Qwen3.5 linear attention/partial RoPE dimensions");
    }
    if (hidden_act != "silu")
        throw std::invalid_argument("Qwen supports hidden_act=silu only");
}

QwenConfig QwenConfig::from_pretrained(const std::string& path) {
    namespace fs = std::filesystem;
    const auto file = fs::is_directory(path) ? fs::path(path) / "config.json" : fs::path(path);
    std::ifstream input(file);
    if (!input)
        throw std::runtime_error("Cannot open config: " + file.string());
    auto j = nlohmann::json::parse(input);
    if (!j.is_object())
        throw std::invalid_argument("Qwen config must be a JSON object");
    QwenConfig cfg;
    cfg.model_type = j.value("model_type", std::string("qwen2"));
    if (cfg.model_type == "qwen3_5") {
        j = j.at("text_config");
        cfg.model_type = j.value("model_type", std::string("qwen3_5_text"));
        cfg.checkpoint_prefix = "model.language_model.";
        if (!cfg.hybrid())
            throw std::invalid_argument("Qwen3.5 wrapper requires qwen3_5_text backbone");
    }
    if (cfg.model_type != "qwen2" && cfg.model_type != "qwen3" && !cfg.hybrid())
        throw std::invalid_argument("Unsupported Qwen model_type: " + cfg.model_type);
    for (const auto* key : {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                            "num_attention_heads", "num_key_value_heads"})
        if (!j.contains(key))
            throw std::invalid_argument(std::string("Qwen config is missing architecture field: ") +
                                        key);
    for (const auto* key :
         {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
          "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
          "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
          "linear_value_head_dim", "linear_conv_kernel_dim", "full_attention_interval"})
        if (j.contains(key) &&
            (!j.at(key).is_number_integer() || j.at(key) < 0 || j.at(key) > INT32_MAX))
            throw std::invalid_argument(std::string("Qwen dimension must be an integer: ") + key);
    if (j.value("attention_dropout", 0.0f) != 0)
        throw std::invalid_argument(
            "Qwen attention dropout is not implemented; use attention_dropout=0");
    cfg.vocab_size = j.value("vocab_size", cfg.vocab_size);
    cfg.hidden_size = j.value("hidden_size", cfg.hidden_size);
    cfg.intermediate_size = j.value("intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = j.value("num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = j.value("num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = j.value("num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim = j.value("head_dim", 0);
    cfg.attention_bias = j.value("attention_bias", cfg.model_type == "qwen2");
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", false);
    cfg.max_position_embeddings = j.value("max_position_embeddings", cfg.max_position_embeddings);
    cfg.rms_norm_eps = j.value("rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta = j.value("rope_theta", cfg.model_type == "qwen3" ? 10000.0f : cfg.rope_theta);
    cfg.hidden_act = j.value("hidden_act", cfg.hidden_act);
    if (cfg.hybrid()) {
        if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null())
            throw std::invalid_argument(
                "Qwen3.5 rope_scaling is not implemented; use default rope_parameters");
        if (!j.value("attn_output_gate", true))
            throw std::invalid_argument("Qwen3.5 requires attn_output_gate=true");
        const auto rope = j.contains("rope_parameters") && !j.at("rope_parameters").is_null()
                              ? j.at("rope_parameters")
                              : nlohmann::json::object();
        if (rope.value("rope_type", std::string("default")) != "default")
            throw std::invalid_argument("Qwen3.5 supports default RoPE only");
        cfg.rope_theta = rope.value("rope_theta", j.value("rope_theta", 10000.0f));
        cfg.partial_rotary_factor =
            rope.value("partial_rotary_factor", j.value("partial_rotary_factor", 0.25f));
        cfg.linear_num_key_heads = j.value("linear_num_key_heads", 16);
        cfg.linear_num_value_heads = j.value("linear_num_value_heads", 32);
        cfg.linear_key_head_dim = j.value("linear_key_head_dim", 128);
        cfg.linear_value_head_dim = j.value("linear_value_head_dim", 128);
        cfg.linear_conv_kernel_dim = j.value("linear_conv_kernel_dim", 4);
        if (j.contains("layer_types"))
            cfg.layer_types = j.at("layer_types").get<std::vector<std::string>>();
        else {
            const int interval = j.value("full_attention_interval", 4);
            if (interval <= 0)
                throw std::invalid_argument("Invalid full_attention_interval");
            for (int i = 0; i < cfg.num_hidden_layers; ++i)
                cfg.layer_types.push_back((i + 1) % interval ? "linear_attention"
                                                             : "full_attention");
        }
    } else {
        if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null())
            throw std::invalid_argument("Qwen non-default rope_scaling is not implemented");
        if (j.contains("rope_parameters") && !j.at("rope_parameters").is_null()) {
            const auto& rope = j.at("rope_parameters");
            if (rope.value("rope_type", std::string("default")) != "default" ||
                rope.value("partial_rotary_factor", 1.0f) != 1.0f)
                throw std::invalid_argument("Qwen supports full-dimension default RoPE only");
            for (const auto& item : rope.items())
                if (item.key() != "rope_type" && item.key() != "rope_theta" &&
                    item.key() != "partial_rotary_factor")
                    throw std::invalid_argument("Unsupported dense Qwen RoPE parameter: " +
                                                item.key());
            cfg.rope_theta = rope.value("rope_theta", cfg.rope_theta);
        }
        if (j.contains("layer_types")) {
            for (const auto& type : j.at("layer_types"))
                if (type != "full_attention")
                    throw std::invalid_argument("Unsupported Qwen layer type");
            if (j.at("layer_types").size() != size_t(cfg.num_hidden_layers))
                throw std::invalid_argument("Qwen layer_types count mismatch");
        }
    }
    if (j.value("use_sliding_window", false))
        throw std::invalid_argument("Qwen sliding-window attention is not implemented");
    auto token = [&](const char* key) {
        return j.contains(key) && j.at(key).is_number_integer() ? j.at(key).get<int>() : -1;
    };
    cfg.bos_token_id = token("bos_token_id");
    cfg.eos_token_id = token("eos_token_id");
    cfg.pad_token_id = token("pad_token_id");
    cfg.validate();
    return cfg;
}

QwenModel::QwenModel(const QwenConfig& cfg) : QwenModel(cfg, WeightInitialization::Allocate) {}

QwenModel::QwenModel(const QwenConfig& cfg, WeightInitialization initialization) : config_(cfg) {
    pretrained_pending_ = initialization == WeightInitialization::Deferred;
    config_.validate();
    config_.head_dim = config_.attention_head_dim();
    auto weight = [initialization](std::vector<int64_t> shape) {
        if (initialization == WeightInitialization::Deferred)
            return std::make_shared<Tensor>(shape, static_cast<void*>(nullptr), kFloat32, kCPU,
                                            true);
        return std::make_shared<Tensor>(shape, kFloat32, kCPU);
    };
    embed_tokens_ = weight({cfg.vocab_size, cfg.hidden_size});
    final_norm_weight_ = weight({cfg.hidden_size});
    if (!cfg.tie_word_embeddings)
        lm_head_weight_ = weight({cfg.vocab_size, cfg.hidden_size});
    layers_.resize(cfg.num_hidden_layers);
    const int64_t qout = int64_t(cfg.num_attention_heads) * config_.head_dim;
    const int64_t kvout = int64_t(cfg.num_key_value_heads) * config_.head_dim;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& b = layers_[i];
        b.linear_attention = config_.linear_layer(i);
        b.input_norm_weight = weight({cfg.hidden_size});
        b.post_norm_weight = weight({cfg.hidden_size});
        if (config_.linear_layer(i)) {
            const int64_t kd = int64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim;
            const int64_t vd = int64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim;
            b.in_proj_qkv_weight = weight({cfg.hidden_size, 2 * kd + vd});
            b.in_proj_z_weight = weight({cfg.hidden_size, vd});
            b.in_proj_a_weight = weight({cfg.hidden_size, cfg.linear_num_value_heads});
            b.in_proj_b_weight = weight({cfg.hidden_size, cfg.linear_num_value_heads});
            b.out_proj_weight = weight({vd, cfg.hidden_size});
            b.conv_weight = weight({2 * kd + vd, 1, cfg.linear_conv_kernel_dim});
            b.log_decay = zeros({cfg.linear_num_value_heads});
            b.dt_bias = zeros({cfg.linear_num_value_heads});
            b.gated_norm_weight = ones({cfg.linear_value_head_dim});
        } else {
            b.q_proj_weight = weight({cfg.hidden_size, qout * (cfg.hybrid() ? 2 : 1)});
            b.k_proj_weight = weight({cfg.hidden_size, kvout});
            b.v_proj_weight = weight({cfg.hidden_size, kvout});
            b.o_proj_weight = weight({qout, cfg.hidden_size});
            if (cfg.attention_bias) {
                b.q_proj_bias = zeros({qout});
                b.k_proj_bias = zeros({kvout});
                b.v_proj_bias = zeros({kvout});
                if (cfg.model_type == "qwen3")
                    b.o_proj_bias = zeros({cfg.hidden_size});
            }
            if (cfg.uses_qk_norm()) {
                b.q_norm_weight = ones({config_.head_dim});
                b.k_norm_weight = ones({config_.head_dim});
            }
        }
        b.gate_proj_weight = weight({cfg.hidden_size, cfg.intermediate_size});
        b.up_proj_weight = weight({cfg.hidden_size, cfg.intermediate_size});
        b.down_proj_weight = weight({cfg.intermediate_size, cfg.hidden_size});
    }
}

std::vector<QwenModel::LinearModule> QwenModel::linear_modules(QwenBlock& b) {
    std::vector<LinearModule> modules;
    if (b.linear_attention)
        modules = {{"in_proj_qkv", "linear_attn", &b.in_proj_qkv_weight, nullptr, &b.in_qkv_lin},
                   {"in_proj_z", "linear_attn", &b.in_proj_z_weight, nullptr, &b.in_z_lin},
                   {"in_proj_a", "linear_attn", &b.in_proj_a_weight, nullptr, &b.in_a_lin},
                   {"in_proj_b", "linear_attn", &b.in_proj_b_weight, nullptr, &b.in_b_lin},
                   {"out_proj", "linear_attn", &b.out_proj_weight, nullptr, &b.out_lin}};
    else
        modules = {{"q_proj", "self_attn", &b.q_proj_weight, &b.q_proj_bias, &b.q_lin},
                   {"v_proj", "self_attn", &b.v_proj_weight, &b.v_proj_bias, &b.v_lin},
                   {"k_proj", "self_attn", &b.k_proj_weight, &b.k_proj_bias, &b.k_lin},
                   {"o_proj", "self_attn", &b.o_proj_weight, &b.o_proj_bias, &b.o_lin}};
    modules.insert(modules.end(),
                   {{"gate_proj", "mlp", &b.gate_proj_weight, nullptr, &b.gate_lin},
                    {"up_proj", "mlp", &b.up_proj_weight, nullptr, &b.up_lin},
                    {"down_proj", "mlp", &b.down_proj_weight, nullptr, &b.down_lin}});
    return modules;
}

std::vector<std::string> QwenModel::available_targets() {
    std::vector<std::string> result;
    for (auto& b : layers_)
        for (const auto& m : linear_modules(b))
            if (std::find(result.begin(), result.end(), m.name) == result.end())
                result.push_back(m.name);
    return result;
}

std::unordered_map<std::string, TensorPtr*> QwenModel::weight_slots() {
    std::unordered_map<std::string, TensorPtr*> slots{{"embed_tokens.weight", &embed_tokens_},
                                                      {"final_norm.weight", &final_norm_weight_}};
    if (lm_head_weight_)
        slots["lm_head.weight"] = &lm_head_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& b = layers_[i];
        const std::string prefix = "layers." + std::to_string(i) + ".";
        slots[prefix + "input_norm.weight"] = &b.input_norm_weight;
        slots[prefix + "post_norm.weight"] = &b.post_norm_weight;
        if (b.q_norm_weight)
            slots[prefix + "self_attn.q_norm.weight"] = &b.q_norm_weight;
        if (b.k_norm_weight)
            slots[prefix + "self_attn.k_norm.weight"] = &b.k_norm_weight;
        if (b.conv_weight) {
            slots[prefix + "linear_attn.conv1d.weight"] = &b.conv_weight;
            slots[prefix + "linear_attn.A_log"] = &b.log_decay;
            slots[prefix + "linear_attn.dt_bias"] = &b.dt_bias;
            slots[prefix + "linear_attn.norm.weight"] = &b.gated_norm_weight;
        }
        for (const auto& module : linear_modules(b)) {
            const auto name = prefix + module.scope + "." + module.name;
            slots[name + ".weight"] = module.weight;
            if (module.bias && *module.bias)
                slots[name + ".bias"] = module.bias;
        }
    }
    return slots;
}

namespace {
void validate_qwen_weight(const std::string& name, const TensorPtr& value,
                          const TensorPtr& expected) {
    if (!value || value->shape() != expected->shape() ||
        !DTypeUtils::is_floating_point(value->dtype()) || !value->device().is_cpu() ||
        (value->ndim() != 2 && value->dtype() != kFloat32))
        throw std::invalid_argument("Invalid shape or dtype for Qwen weight: " + name);
}
} // namespace

void QwenModel::assign_weight(const std::string& key, const TensorPtr& tensor) {
    if (offload_ || lora_rank_)
        throw std::logic_error("load base weights before LoRA/offload");
    auto slots = weight_slots();
    auto found = slots.find(key);
    if (found == slots.end())
        throw std::invalid_argument("Unexpected Qwen weight: " + key);
    validate_qwen_weight(key, tensor, *found->second);
    *found->second = tensor;
}

void QwenModel::load_pretrained(const std::string& directory,
                                const SafeTensorsLoadOptions& options) {
    if (lora_rank_ || offload_)
        throw std::logic_error("load pretrained weights before LoRA/offload");
    SafeTensorsModelReader reader(directory);
    reader.parse_headers();
    auto names = reader.get_tensor_names();
    std::unordered_set<std::string> available(names.begin(), names.end());
    auto checkpoint_config = config_;
    if (config_.hybrid()) {
        const bool plain = available.count("model.embed_tokens.weight");
        const bool nested = available.count("model.language_model.embed_tokens.weight");
        if (plain == nested)
            throw std::invalid_argument(
                "Qwen3.5 checkpoint must have one unambiguous text backbone");
        checkpoint_config.checkpoint_prefix = nested ? "model.language_model." : "model.";
    }
    auto mapping = QwenKeyMapper::generate_qwen_mapping(checkpoint_config);
    for (const auto& item : mapping)
        if (!available.count(item.second))
            throw std::runtime_error("Missing required Qwen pretrained weight: " + item.second);
    auto opts = options;
    opts.transpose_linear = true;
    opts.skip_transpose_keys.push_back("lm_head.weight");
    // Convolution and normalization kernels consume FP32 base weights.
    auto tensors = reader.load_tensors_mapped(mapping, opts);
    for (auto& item : tensors)
        if (item.second->ndim() != 2 && item.second->dtype() != kFloat32) {
            auto converted = zeros(item.second->shape());
            for (int64_t i = 0; i < converted->numel(); ++i)
                converted->data<float>()[i] = qwen_weight_value(item.second, i);
            item.second = converted;
        }
    auto slots = weight_slots();
    if (tensors.size() != slots.size())
        throw std::runtime_error("Qwen pretrained tensor count does not match architecture");
    // Check every tensor before replacing any model-owned weight.
    for (const auto& item : tensors)
        validate_qwen_weight(item.first, item.second, *slots.at(item.first));
    for (const auto& item : tensors)
        *slots.at(item.first) = item.second;
    pretrained_pending_ = false;
}

// -------------- LoRA --------------
void QwenModel::init_lora(int rank, float alpha, float dropout, bool qv_only, uint64_t seed) {
    init_lora(rank, alpha, dropout,
              qv_only ? std::vector<std::string>{"q_proj", "v_proj"} : std::vector<std::string>{},
              seed);
}

void QwenModel::init_lora(int rank, float alpha, float dropout,
                          const std::vector<std::string>& requested, uint64_t seed) {
    if (pretrained_pending_)
        throw std::logic_error(
            "Complete load_pretrained before initializing a deferred Qwen model");
    if (offload_)
        throw std::logic_error("initialize LoRA before enabling parameter offload");
    if (lora_rank_)
        throw std::logic_error("Qwen LoRA is already initialized");
    if (rank <= 0 || !std::isfinite(alpha) || alpha <= 0 || !std::isfinite(dropout) ||
        dropout < 0 || dropout >= 1)
        throw std::invalid_argument("LoRA requires positive rank/alpha and dropout in [0,1)");
    if (activation_checkpointing_ && dropout > 0)
        throw std::invalid_argument("Qwen checkpointing requires zero LoRA dropout");
    auto allowed_list = available_targets();
    auto targets = requested;
    if (targets.empty())
        targets = config_.default_lora_targets();
    if (targets == std::vector<std::string>{"all-linear"})
        targets = allowed_list;
    std::unordered_set<std::string> unique;
    const std::unordered_set<std::string> allowed(allowed_list.begin(), allowed_list.end());
    for (const auto& name : targets)
        if (!allowed.count(name) || !unique.insert(name).second)
            throw std::invalid_argument("Unknown or duplicate Qwen LoRA target: " + name);
    // Validate every layer first; dimensions come from actual loaded matrices.
    for (auto& b : layers_)
        for (const auto& module : linear_modules(b))
            if (unique.count(module.name)) {
                const auto& w = *module.weight;
                if (!w || w->ndim() != 2)
                    throw std::invalid_argument("Qwen LoRA target is not a loaded linear weight");
            }
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> distribution(-0.01f, 0.01f);
    for (auto& b : layers_) {
        for (const auto& module : linear_modules(b))
            if (unique.count(module.name)) {
                const auto& w = *module.weight;
                auto a = zeros({w->shape()[0], rank});
                auto bt = zeros({rank, w->shape()[1]});
                for (int64_t i = 0; i < a->numel(); ++i)
                    a->data<float>()[i] = distribution(rng);
                *module.adapter =
                    std::make_unique<LoRALinear>(w, module.bias ? *module.bias : nullptr);
                (*module.adapter)->attach_lora(a, bt, alpha / rank);
                (*module.adapter)->set_dropout(dropout, seed++);
                (*module.adapter)->set_training(training_);
            }
        b.lora_initialized = true;
    }
    lora_rank_ = rank;
    lora_alpha_ = alpha;
    lora_dropout_ = dropout;
    lora_targets_ = targets;
}

std::vector<TensorPtr> QwenModel::get_lora_parameters() const {
    std::vector<TensorPtr> params;
    for (const auto& b : layers_)
        for (auto* lin : {b.q_lin.get(), b.v_lin.get(), b.k_lin.get(), b.o_lin.get(),
                          b.gate_lin.get(), b.up_lin.get(), b.down_lin.get(), b.in_qkv_lin.get(),
                          b.in_z_lin.get(), b.in_a_lin.get(), b.in_b_lin.get(), b.out_lin.get()})
            if (lin) {
                auto p = lin->trainable_parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
    return params;
}

void QwenModel::set_training(bool training) {
    training_ = training;
    for (auto& b : layers_)
        for (const auto& module : linear_modules(b))
            if (*module.adapter)
                (*module.adapter)->set_training(training);
}

std::vector<TensorPtr> QwenModel::parameters() const {
    if (offload_)
        throw std::logic_error("use get_lora_parameters after enabling frozen base offload");
    if (config_.hybrid()) {
        std::vector<TensorPtr> params;
        for (const auto& slot : const_cast<QwenModel*>(this)->weight_slots())
            params.push_back(*slot.second);
        return params;
    }
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
    if (lm_head_weight_)
        params.push_back(lm_head_weight_);
    for (const auto& b : layers_) {
        if (b.q_norm_weight)
            params.push_back(b.q_norm_weight);
        if (b.k_norm_weight)
            params.push_back(b.k_norm_weight);
        if (b.o_proj_bias)
            params.push_back(b.o_proj_bias);
    }
    params.erase(std::remove(params.begin(), params.end(), nullptr), params.end());
    return params;
}

void QwenModel::freeze_base() {
    for (auto& p : parameters())
        p->set_requires_grad(false);
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
            if (id < 0 || id >= config_.vocab_size)
                throw std::invalid_argument("Qwen input token out of vocabulary range");
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

TensorPtr QwenModel::attention(const TensorPtr& x, QwenBlock& blk, const TensorPtr& causal_mask,
                               const TensorPtr& pad_mask, int64_t seq_len [[maybe_unused]]) {
    int64_t B = x->shape()[0];
    int64_t S = x->shape()[1];
    int64_t H = config_.num_attention_heads;
    int64_t HKV = config_.num_key_value_heads;
    int64_t Hd = config_.head_dim;
    // int64_t kv_out = HKV * Hd;

    auto linear = [&](const TensorPtr& w, const TensorPtr& bias,
                      const std::unique_ptr<LoRALinear>& lora) {
        if (lora)
            return lora->forward(x);
        auto y = matmul(x, w);
        return bias ? add(y, bias) : y;
    };
    auto q = linear(blk.q_proj_weight, blk.q_proj_bias, blk.q_lin);
    auto k = linear(blk.k_proj_weight, blk.k_proj_bias, blk.k_lin);
    auto v = linear(blk.v_proj_weight, blk.v_proj_bias, blk.v_lin);

    TensorPtr gate;
    if (config_.hybrid()) {
        q = reshape(q, {B, S, H, 2 * Hd});
        gate = reshape(q->slice(3, Hd, 2 * Hd), {B, S, H * Hd});
        q = q->slice(3, 0, Hd);
    }
    q = reshape(q, {B, S, H, Hd});
    if (blk.q_norm_weight)
        q = norm(q, blk.q_norm_weight);
    q = permute(q, {0, 2, 1, 3}); // [B,H,S,Hd]

    k = reshape(k, {B, S, HKV, Hd});
    v = reshape(v, {B, S, HKV, Hd});
    if (blk.k_norm_weight)
        k = norm(k, blk.k_norm_weight);
    k = permute(k, {0, 2, 1, 3}); // [B,HKV,S,Hd]
    v = permute(v, {0, 2, 1, 3});

    // GQA: repeat kv heads to match H
    if (HKV != H) {
        int repeat = H / HKV;
        k = repeat_kv_heads(k, repeat); // [B,H,S,Hd]
        v = repeat_kv_heads(v, repeat);
    }

    if (config_.hybrid()) {
        q = partial_rotary_embedding(q, int(Hd * config_.partial_rotary_factor),
                                     config_.rope_theta);
        k = partial_rotary_embedding(k, int(Hd * config_.partial_rotary_factor),
                                     config_.rope_theta);
    } else {
        q = apply_rope(q, S, Hd, config_.rope_theta);
        k = apply_rope(k, S, Hd, config_.rope_theta);
    }

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

        if (causal_mask)
            scores = add(scores, causal_mask);
        if (pad_mask)
            scores = add(scores, pad_mask);
        auto probs = softmax(scores, -1);
        ctx = matmul(probs, v); // [B,H,S,Hd]
    }
    ctx = permute(ctx, {0, 2, 1, 3});
    ctx = reshape(ctx, {B, S, H * Hd});

    if (gate)
        ctx = mul(ctx, sigmoid(gate));
    auto out = blk.o_lin ? blk.o_lin->forward(ctx) : matmul(ctx, blk.o_proj_weight);
    if (!blk.o_lin && blk.o_proj_bias)
        out = add(out, blk.o_proj_bias);
    return out;
}

TensorPtr QwenModel::norm(const TensorPtr& x, const TensorPtr& w) {
    return rms_norm_affine(x, config_.hybrid() ? add(ones(w->shape()), w) : w,
                           config_.rms_norm_eps);
}

TensorPtr QwenModel::linear_attention(const TensorPtr& input, QwenBlock& b, const TensorPtr& mask) {
    auto x = input;
    const int64_t B = x->shape()[0], S = x->shape()[1];
    // Match the reference's padding convention for batched full-sequence
    // training.
    if (mask && B > 1 && S > 1)
        x = mul(x, reshape(mask, {B, S, 1}));
    auto linear = [&](const TensorPtr& w, const std::unique_ptr<LoRALinear>& l) {
        return l ? l->forward(x) : matmul(x, w);
    };
    const int64_t HK = config_.linear_num_key_heads, HV = config_.linear_num_value_heads;
    const int64_t K = config_.linear_key_head_dim, V = config_.linear_value_head_dim;
    auto qkv =
        silu(causal_depthwise_conv1d(linear(b.in_proj_qkv_weight, b.in_qkv_lin), b.conv_weight));
    auto q = reshape(qkv->slice(2, 0, HK * K), {B, S, HK, K});
    auto k = reshape(qkv->slice(2, HK * K, 2 * HK * K), {B, S, HK, K});
    auto v = reshape(qkv->slice(2, 2 * HK * K, 2 * HK * K + HV * V), {B, S, HV, V});
    auto y = gated_delta_rule(q, k, v, linear(b.in_proj_a_weight, b.in_a_lin),
                              linear(b.in_proj_b_weight, b.in_b_lin), b.log_decay, b.dt_bias);
    auto z = reshape(linear(b.in_proj_z_weight, b.in_z_lin), {B, S, HV, V});
    y = mul(rms_norm_affine(y, b.gated_norm_weight, config_.rms_norm_eps), silu(z));
    y = reshape(y, {B, S, HV * V});
    return b.out_lin ? b.out_lin->forward(y) : matmul(y, b.out_proj_weight);
}

TensorPtr QwenModel::mlp(const TensorPtr& x, QwenBlock& blk) {
    auto gate = blk.gate_lin ? blk.gate_lin->forward(x) : matmul(x, blk.gate_proj_weight);
    auto up = blk.up_lin ? blk.up_lin->forward(x) : matmul(x, blk.up_proj_weight);
    auto act = swiglu(gate, up);
    auto down = blk.down_lin ? blk.down_lin->forward(act) : matmul(act, blk.down_proj_weight);
    return down;
}

// -------------- 前向 --------------
TensorPtr QwenModel::forward_hidden(const TensorPtr& input_ids, const TensorPtr& attention_mask) {
    if (pretrained_pending_)
        throw std::logic_error("Complete load_pretrained before using a deferred Qwen model");
    if (!input_ids || input_ids->dtype() != kInt32 || input_ids->ndim() != 2 ||
        input_ids->shape()[0] <= 0 || input_ids->shape()[1] <= 0 ||
        input_ids->shape()[1] > config_.max_position_embeddings)
        throw std::invalid_argument("Qwen input_ids must be nonempty int32 "
                                    "[batch,seq] within max_position_embeddings");
    if (attention_mask &&
        (attention_mask->shape() != input_ids->shape() || attention_mask->dtype() != kFloat32))
        throw std::invalid_argument("Qwen attention_mask must be float32 and match input_ids");
    if (offload_)
        offload_->load(0);
    auto x = embedding_lookup(embed_tokens_, input_ids); // [B,S,H]
    if (offload_)
        offload_->offload_all();

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
        double sum = 0.0, sumsq = 0.0;
        int64_t n = t->numel();
        for (int64_t i = 0; i < n; ++i) {
            sum += p[i];
            sumsq += p[i] * p[i];
        }
        double mean = sum / n;
        double std = std::sqrt(sumsq / n - mean * mean);
        std::cout << name << " mean " << mean << " std " << std << std::endl;
    };

    auto causal_mask = build_causal_mask(x->shape()[1]);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        auto block_forward = [this, i, causal_mask, pad_mask, attention_mask, debug,
                              tensor_stats](const TensorPtr& input) {
            if (offload_)
                offload_->load(static_cast<size_t>(i) + 2);
            auto& blk = layers_[i];
            OffloadLinearLease lease{bool(offload_)};
            for (const auto& module : linear_modules(blk))
                lease.bind(module.adapter->get(), *module.weight,
                           module.bias ? *module.bias : nullptr);
            auto x = input;
            auto normed = norm(x, blk.input_norm_weight);
            auto attn_out = config_.linear_layer(i)
                                ? linear_attention(normed, blk, attention_mask)
                                : attention(normed, blk, causal_mask, pad_mask, x->shape()[1]);
            x = add(x, attn_out);
            auto normed2 = norm(x, blk.post_norm_weight);
            auto mlp_out = mlp(normed2, blk);
            x = add(x, mlp_out);

            if (debug && i == 0) {
                tensor_stats(normed, "ln1");
                tensor_stats(attn_out, "attn_out");
                tensor_stats(normed2, "ln2");
                tensor_stats(mlp_out, "mlp_out");
            }
            return x;
        };
        std::vector<TensorPtr> captures;
        for (const auto& module : linear_modules(layers_[i]))
            if (*module.adapter) {
                auto params = (*module.adapter)->trainable_parameters();
                captures.insert(captures.end(), params.begin(), params.end());
            }
        x = activation_checkpointing_ ? checkpoint(block_forward, x, captures) : block_forward(x);
        if (offload_)
            offload_->offload_all();
    }
    if (offload_)
        offload_->load(1);
    x = norm(x, final_norm_weight_);
    if (offload_)
        offload_->offload_all();
    return x;
}

TensorPtr QwenModel::lm_head(const TensorPtr& hidden) {
    if (offload_)
        offload_->load(0);
    auto output = matmul_rhs_T(hidden, output_weight());
    if (offload_)
        offload_->offload_all();
    return output;
}

TensorPtr QwenModel::loss_from_hidden(const TensorPtr& hidden, const TensorPtr& labels,
                                      bool streamed, int ignore_index) {
    if (!streamed)
        return lm_cross_entropy(lm_head(hidden), labels, ignore_index, "mean");
    if (offload_)
        offload_->load(0);
    try {
        auto loss =
            streaming_lm_cross_entropy(hidden, output_weight(), labels, ignore_index, "mean");
        if (offload_)
            offload_->offload_all();
        return loss;
    } catch (...) {
        if (offload_)
            offload_->offload_all();
        throw;
    }
}

TensorPtr QwenModel::forward(const TensorPtr& input_ids, const TensorPtr& attention_mask) {
    return lm_head(forward_hidden(input_ids, attention_mask));
}

void QwenModel::set_activation_checkpointing(bool enabled) {
#ifndef USE_NEW_AUTOGRAD_ENGINE
    if (enabled)
        throw std::logic_error("activation checkpointing requires the new autograd engine");
#endif
    if (enabled && lora_dropout_ > 0)
        throw std::invalid_argument("Qwen checkpointing requires zero LoRA dropout");
    activation_checkpointing_ = enabled;
}

void QwenModel::enable_parameter_sharding(const sharding::ShardConfig& config) {
#ifndef USE_NEW_AUTOGRAD_ENGINE
    (void)config;
    throw std::logic_error("parameter offload requires the new autograd engine");
#else
    if (offload_)
        throw std::logic_error("parameter offload already enabled");
    for (auto& block : layers_) {
        if (!block.lora_initialized)
            throw std::logic_error("initialize LoRA before offload");
        for (const auto& module : linear_modules(block))
            if (*module.adapter && (*module.adapter)->is_merged())
                throw std::logic_error("unmerge LoRA before offload");
    }
    // Loading, validation and offload share one complete weight inventory.
    // Group 0: embedding/head; 1: final norm; remaining groups: whole blocks.
    std::vector<FrozenModelOffload::Group> groups(layers_.size() + 2);
    auto slots = weight_slots();
    std::vector<std::string> names;
    for (const auto& item : slots)
        names.push_back(item.first);
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        size_t group = 0;
        if (name == "final_norm.weight")
            group = 1;
        else if (name.rfind("layers.", 0) == 0)
            group = static_cast<size_t>(std::stoul(name.substr(7))) + 2;
        groups.at(group).push_back(slots.at(name));
    }
    offload_ = std::make_unique<FrozenModelOffload>(config, groups);
    for (auto& block : layers_)
        for (const auto& module : linear_modules(block))
            if (*module.adapter)
                (*module.adapter)->set_base_weights(nullptr);
    offload_->offload_all();
#endif
}

} // namespace ops
