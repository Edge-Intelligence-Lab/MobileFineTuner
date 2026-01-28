#include "gemma_model.h"

#include "../core/ops.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

std::string extract_string(const std::string& content, const std::string& key, const std::string& def = "") {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*\"([^\"]+)\")";
    std::regex pattern(pattern_str);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str();
    }
    return def;
}

int extract_int(const std::string& content, const std::string& key, int def) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*(-?\d+))";
    std::regex pattern(pattern_str);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return def;
}

float extract_float(const std::string& content, const std::string& key, float def) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*(-?\d+(\.\d+)?([eE][-+]?\d+)?))";
    std::regex pattern(pattern_str);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return std::stof(match[1].str());
    }
    return def;
}

bool extract_bool(const std::string& content, const std::string& key, bool def) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*(true|false))";
    std::regex pattern(pattern_str, std::regex::icase);
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        std::string v = match[1].str();
        for (auto& ch : v) ch = static_cast<char>(std::tolower(ch));
        return v == "true";
    }
    return def;
}

std::vector<std::string> extract_string_array(const std::string& content, const std::string& key) {
    std::string pattern_str = "\\\"" + key + R"(\"\s*:\s*\[([^\]]*)\])";
    std::regex pattern(pattern_str);
    std::smatch match;
    std::vector<std::string> result;
    if (std::regex_search(content, match, pattern)) {
        std::string arr = match[1].str();
        std::regex item_pattern(R"(\"([^\"]+)\")");
        auto it_begin = std::sregex_iterator(arr.begin(), arr.end(), item_pattern);
        auto it_end = std::sregex_iterator();
        for (auto it = it_begin; it != it_end; ++it) {
            result.push_back((*it)[1].str());
        }
    }
    return result;
}

}  // namespace

namespace {

enum class DumpDType { kFloat32, kInt32 };

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

namespace ops {

GemmaTextConfig GemmaTextConfig::from_pretrained(const std::string& model_dir) {
    GemmaTextConfig cfg;
    const std::string path = model_dir + "/config.json";
    const std::string content = read_file(path);

    cfg.vocab_size = extract_int(content, "vocab_size", cfg.vocab_size);
    cfg.hidden_size = extract_int(content, "hidden_size", cfg.hidden_size);
    cfg.intermediate_size = extract_int(content, "intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = extract_int(content, "num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = extract_int(content, "num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = extract_int(content, "num_key_value_heads", cfg.num_key_value_heads);
    cfg.head_dim = extract_int(content, "head_dim", cfg.head_dim);
    cfg.max_position_embeddings = extract_int(content, "max_position_embeddings", cfg.max_position_embeddings);
    cfg.sliding_window = extract_int(content, "sliding_window", cfg.sliding_window);
    cfg.attention_bias = extract_bool(content, "attention_bias", cfg.attention_bias);
    cfg.use_bidirectional_attention =
        extract_bool(content, "use_bidirectional_attention", cfg.use_bidirectional_attention);
    cfg.use_cache = extract_bool(content, "use_cache", cfg.use_cache);

    cfg.attention_dropout = extract_float(content, "attention_dropout", cfg.attention_dropout);
    cfg.rms_norm_eps = extract_float(content, "rms_norm_eps", cfg.rms_norm_eps);
    cfg.query_pre_attn_scalar = extract_float(content, "query_pre_attn_scalar", cfg.query_pre_attn_scalar);
    cfg.attn_logit_softcapping = extract_float(content, "attn_logit_softcapping", cfg.attn_logit_softcapping);
    cfg.final_logit_softcapping = extract_float(content, "final_logit_softcapping", cfg.final_logit_softcapping);
    cfg.rope_theta = extract_float(content, "rope_theta", cfg.rope_theta);
    cfg.rope_local_base_freq = extract_float(content, "rope_local_base_freq", cfg.rope_local_base_freq);
    cfg.hidden_activation = extract_string(content, "hidden_activation", cfg.hidden_activation);

    cfg.layer_types = extract_string_array(content, "layer_types");
    if (cfg.layer_types.empty()) {
        cfg.layer_types.assign(cfg.num_hidden_layers, "sliding_attention");
    }
    return cfg;
}

namespace {

constexpr float kMaskValue = -1e10f;

}  // namespace

GemmaModel::GemmaModel(const GemmaTextConfig& config)
    : config_(config) {
    embed_weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{config_.vocab_size, config_.hidden_size}, DType::kFloat32, kCPU);
    norm_weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{config_.hidden_size}, DType::kFloat32, kCPU);
    lm_head_weight_ = std::make_shared<Tensor>(
        std::vector<int64_t>{config_.hidden_size, config_.vocab_size}, DType::kFloat32, kCPU);

    blocks_.resize(config_.num_hidden_layers);
    for (auto& block : blocks_) {
        block.input_layernorm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size}, DType::kFloat32, kCPU);
        block.post_attention_layernorm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size}, DType::kFloat32, kCPU);
        block.pre_feedforward_layernorm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size}, DType::kFloat32, kCPU);
        block.post_feedforward_layernorm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size}, DType::kFloat32, kCPU);

        block.q_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size, config_.num_attention_heads * config_.head_dim}, DType::kFloat32, kCPU);
        block.k_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size, config_.num_key_value_heads * config_.head_dim}, DType::kFloat32, kCPU);
        block.v_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size, config_.num_key_value_heads * config_.head_dim}, DType::kFloat32, kCPU);
        block.o_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.num_attention_heads * config_.head_dim, config_.hidden_size}, DType::kFloat32, kCPU);

        block.q_norm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.head_dim}, DType::kFloat32, kCPU);
        block.k_norm_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.head_dim}, DType::kFloat32, kCPU);

        block.gate_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size, config_.intermediate_size}, DType::kFloat32, kCPU);
        block.up_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.hidden_size, config_.intermediate_size}, DType::kFloat32, kCPU);
        block.down_proj_weight = std::make_shared<Tensor>(
            std::vector<int64_t>{config_.intermediate_size, config_.hidden_size}, DType::kFloat32, kCPU);
    }
}

TensorPtr GemmaModel::embedding_lookup(const TensorPtr& indices) const {
    const auto& idx_shape = indices->shape();
    if (idx_shape.size() != 2) {
        throw std::runtime_error("GemmaModel::embedding_lookup expects [batch, seq_len]");
    }
    int64_t batch = idx_shape[0];
    int64_t seq_len = idx_shape[1];
    auto result = zeros({batch, seq_len, config_.hidden_size}, DType::kFloat32, kCPU);

    const float* emb_data = embed_weight_->data<float>();
    float* out_data = result->data<float>();

    float scale = std::sqrt(static_cast<float>(config_.hidden_size));
    auto copy_row = [&](int32_t token_id, float* dst) {
        if (token_id < 0 || token_id >= config_.vocab_size) {
            throw std::runtime_error("Token id out of range in Gemma embedding");
        }
        const float* src = emb_data + token_id * config_.hidden_size;
        for (int64_t i = 0; i < config_.hidden_size; ++i) {
            dst[i] = src[i] * scale;
        }
    };

    if (indices->dtype() == DType::kInt32) {
        const int32_t* ids = indices->data<int32_t>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int32_t token = ids[b * seq_len + s];
                float* dst = out_data + (b * seq_len + s) * config_.hidden_size;
                copy_row(token, dst);
            }
        }
    } else {
        throw std::runtime_error("GemmaModel::embedding_lookup expects int32 input_ids");
    }

    return result;
}

TensorPtr GemmaModel::build_causal_mask(int seq_len) const {
    auto mask = full({seq_len, seq_len}, 0.0f, DType::kFloat32, kCPU);
    float* data = mask->data<float>();
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = i + 1; j < seq_len; ++j) {
            data[i * seq_len + j] = kMaskValue;
        }
    }
    return mask;
}

TensorPtr GemmaModel::build_sliding_mask(int seq_len) const {
    auto mask = full({seq_len, seq_len}, 0.0f, DType::kFloat32, kCPU);
    float* data = mask->data<float>();
    int window = config_.sliding_window;
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
            bool allow = (j <= i) && (i - j < window);
            if (!allow) {
                data[i * seq_len + j] = kMaskValue;
            }
        }
    }
    return mask;
}

TensorPtr GemmaModel::build_padding_mask(const TensorPtr& attention_mask) const {
    if (!attention_mask) return nullptr;
    const auto& shape = attention_mask->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Gemma attention_mask must be [batch, seq_len]");
    }
    int64_t batch = shape[0];
    int64_t seq_len = shape[1];
    auto mask = zeros({batch, 1, 1, seq_len}, DType::kFloat32, kCPU);
    float* mask_data = mask->data<float>();

    if (attention_mask->dtype() == DType::kFloat32) {
        const float* src = attention_mask->data<float>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                if (src[b * seq_len + s] <= 0.5f) {
                    mask_data[b * seq_len + s] = kMaskValue;
                }
            }
        }
    } else if (attention_mask->dtype() == DType::kInt32) {
        const int32_t* src = attention_mask->data<int32_t>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                if (src[b * seq_len + s] == 0) {
                    mask_data[b * seq_len + s] = kMaskValue;
                }
            }
        }
    } else {
        throw std::runtime_error("Gemma attention_mask must be int32 or float32");
    }

    return mask;
}

GemmaModel::RotaryCache GemmaModel::build_rotary_embeddings(int batch,
                                                            int seq_len,
                                                            float theta) const {
    auto cos = zeros({batch, seq_len, config_.head_dim}, DType::kFloat32, kCPU);
    auto sin = zeros({batch, seq_len, config_.head_dim}, DType::kFloat32, kCPU);
    float* cos_data = cos->data<float>();
    float* sin_data = sin->data<float>();

    int64_t half = config_.head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int64_t i = 0; i < half; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(config_.head_dim);
        inv_freq[i] = std::pow(theta, -exponent);
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            float pos = static_cast<float>(s);
            float* cos_row = cos_data + (b * seq_len + s) * config_.head_dim;
            float* sin_row = sin_data + (b * seq_len + s) * config_.head_dim;
            for (int64_t i = 0; i < half; ++i) {
                float angle = pos * inv_freq[i];
                float c = std::cos(angle);
                float sn = std::sin(angle);
                cos_row[i] = c;
                cos_row[i + half] = c;
                sin_row[i] = sn;
                sin_row[i + half] = sn;
            }
        }
    }

    return {cos, sin};
}

TensorPtr GemmaModel::apply_attention(const TensorPtr& x,
                                      GemmaBlockWeights& block,
                                      const TensorPtr& position_cos,
                                      const TensorPtr& position_sin,
                                      const TensorPtr& pad_mask,
                                      const TensorPtr& base_mask,
                                      float rope_theta,
                                      int dbg_layer) const {
    (void)position_cos;
    (void)position_sin;
    int64_t B = x->shape()[0];
    int64_t S = x->shape()[1];
    int64_t n_head = config_.num_attention_heads;
    int64_t kv_heads = config_.num_key_value_heads;
    int64_t Hd = config_.head_dim;

    auto linear_forward = [&](const TensorPtr& input,
                              const std::unique_ptr<LoRALinear>& linear,
                              const TensorPtr& weight) -> TensorPtr {
        if (linear) {
            return linear->forward(input);
        }
        return matmul(input, weight);
    };

    auto q = linear_forward(x, block.q_proj_lora, block.q_proj_weight);
    auto k = linear_forward(x, block.k_proj_lora, block.k_proj_weight);
    auto v = linear_forward(x, block.v_proj_lora, block.v_proj_weight);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(q, "q_proj_out_l" + std::to_string(dbg_layer));
        dump_tensor(k, "k_proj_out_l" + std::to_string(dbg_layer));
        dump_tensor(v, "v_proj_out_l" + std::to_string(dbg_layer));
    }
    // Allow numeric perturbation exactly at pre-norm projection outputs
    auto perturb_if_match = [&](const std::string& name, const TensorPtr& t) {
        if (debug_.numeric_enabled && debug_.numeric_name == name &&
            debug_.numeric_index >= 0 && debug_.numeric_index < t->numel()) {
            float* data = t->data<float>();
            data[debug_.numeric_index] += debug_.numeric_eps;
        }
    };
    perturb_if_match("q_proj_out_l" + std::to_string(dbg_layer), q);
    perturb_if_match("k_proj_out_l" + std::to_string(dbg_layer), k);
    perturb_if_match("v_proj_out_l" + std::to_string(dbg_layer), v);

    q = reshape(q, {B, S, n_head, Hd});
    k = reshape(k, {B, S, kv_heads, Hd});
    v = reshape(v, {B, S, kv_heads, Hd});

    q = permute(q, {0, 2, 1, 3});  // [B,n_head,S,Hd]
    k = permute(k, {0, 2, 1, 3});
    v = permute(v, {0, 2, 1, 3});

    // Optional: allow disabling q/k RMSNorm via env for isolation tests
    const char* disable_qk_norm = std::getenv("DISABLE_QK_NORM");
      if (!(disable_qk_norm && std::string(disable_qk_norm) == "1")) {
          // Debug: dump inv_rms before applying RMSNorm to pinpoint reduction/broadcast differences
          if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
              // Manually compute inv_rms to avoid interference from generic reduce paths
              // q,k: [B,H,S,Hd] -> inv_rms: [B,H,S,1]
              {
                  auto inv = zeros({B, n_head, S, 1}, DType::kFloat32, kCPU);
                  const float* q_data = q->data<float>();
                  float* inv_data = inv->data<float>();
                  for (int64_t b = 0; b < B; ++b) {
                      for (int64_t h = 0; h < n_head; ++h) {
                          for (int64_t s = 0; s < S; ++s) {
                              int64_t base = (((b * n_head) + h) * S + s) * Hd;
                              double sqsum = 0.0;
                              for (int64_t d = 0; d < Hd; ++d) {
                                  float v = q_data[base + d];
                                  sqsum += static_cast<double>(v) * static_cast<double>(v);
                              }
                              float inv_rms = static_cast<float>(1.0 / std::sqrt(sqsum / static_cast<double>(Hd) + static_cast<double>(config_.rms_norm_eps)));
                              inv_data[((b * n_head) + h) * S + s] = inv_rms;
                          }
                      }
                  }
                  dump_tensor(inv, "q_inv_rms_l" + std::to_string(dbg_layer));
              }
              {
                  auto inv = zeros({B, kv_heads, S, 1}, DType::kFloat32, kCPU);
                  const float* k_data = k->data<float>();
                  float* inv_data = inv->data<float>();
                  for (int64_t b = 0; b < B; ++b) {
                      for (int64_t h = 0; h < kv_heads; ++h) {
                          for (int64_t s = 0; s < S; ++s) {
                              int64_t base = (((b * kv_heads) + h) * S + s) * Hd;
                              double sqsum = 0.0;
                              for (int64_t d = 0; d < Hd; ++d) {
                                  float v = k_data[base + d];
                                  sqsum += static_cast<double>(v) * static_cast<double>(v);
                              }
                              float inv_rms = static_cast<float>(1.0 / std::sqrt(sqsum / static_cast<double>(Hd) + static_cast<double>(config_.rms_norm_eps)));
                              inv_data[((b * kv_heads) + h) * S + s] = inv_rms;
                          }
                      }
                  }
                  dump_tensor(inv, "k_inv_rms_l" + std::to_string(dbg_layer));
              }
          }
          q = rms_norm(q, block.q_norm_weight, config_.rms_norm_eps);
          k = rms_norm(k, block.k_norm_weight, config_.rms_norm_eps);
      }
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(q, "q_norm_out_l" + std::to_string(dbg_layer));
        dump_tensor(k, "k_norm_out_l" + std::to_string(dbg_layer));
        dump_tensor(q, "q_norm_out_l" + std::to_string(dbg_layer) + "_pre_rope");
        dump_tensor(k, "k_norm_out_l" + std::to_string(dbg_layer) + "_pre_rope");
    }

    // apply rotary embeddings with tracked autograd
    q = apply_rope(q, S, Hd, rope_theta);
    k = apply_rope(k, S, Hd, rope_theta);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(q, "q_rotary_out_l" + std::to_string(dbg_layer));
        dump_tensor(k, "k_rotary_out_l" + std::to_string(dbg_layer));
    }

    // Numeric gradient perturbation: allow targeting normalized q/k before scores
    perturb_if_match("q_norm_out_l" + std::to_string(dbg_layer), q);
    perturb_if_match("k_norm_out_l" + std::to_string(dbg_layer), k);

    auto k_full = repeat_kv_heads(k, n_head / kv_heads);
    auto v_full = repeat_kv_heads(v, n_head / kv_heads);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(v_full, "v_full_l" + std::to_string(dbg_layer));
    }

    auto k_t = transpose(k_full, 2, 3);  // [B,n_head,Hd,S]
    auto scores = matmul(q, k_t);        // [B,n_head,S,S]

    float scaling = std::pow(config_.query_pre_attn_scalar, -0.5f);
    scores = mul(scores, scaling);

    if (base_mask) {
        scores = add(scores, base_mask);
    }
    if (pad_mask) {
        scores = add(scores, pad_mask);
    }

    auto probs = softmax(scores, -1);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(scores, "attn_scores_l" + std::to_string(dbg_layer));
        dump_tensor(probs, "attn_probs_l" + std::to_string(dbg_layer));
    }
    auto context = matmul(probs, v_full);  // [B,n_head,S,Hd]
    context = permute(context, {0, 2, 1, 3});
    int64_t attn_dim = block.q_proj_weight->shape()[1];
    context = reshape(context, {B, S, attn_dim});
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(context, "attn_context_l" + std::to_string(dbg_layer));
    }
    perturb_if_match("attn_context_l" + std::to_string(dbg_layer), context);

    auto attn_out = linear_forward(context, block.o_proj_lora, block.o_proj_weight);
    perturb_if_match("attn_out_raw_l" + std::to_string(dbg_layer), attn_out);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(attn_out, "attn_out_raw_l" + std::to_string(dbg_layer));
    }
    return attn_out;
}

TensorPtr GemmaModel::apply_mlp(const TensorPtr& x,
                                GemmaBlockWeights& block,
                                int dbg_layer) const {
    auto linear_forward = [&](const TensorPtr& input,
                              const std::unique_ptr<LoRALinear>& linear,
                              const TensorPtr& weight) -> TensorPtr {
        if (linear) {
            return linear->forward(input);
        }
        return matmul(input, weight);
    };

    auto gate = linear_forward(x, block.gate_proj_lora, block.gate_proj_weight);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(gate, "gate_proj_out_l" + std::to_string(dbg_layer));
    }
    auto gate_act = gelu(gate);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(gate_act, "gate_act_l" + std::to_string(dbg_layer));
    }
    auto up = linear_forward(x, block.up_proj_lora, block.up_proj_weight);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(up, "up_proj_out_l" + std::to_string(dbg_layer));
    }
    auto prod = mul(gate_act, up);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(prod, "mlp_prod_l" + std::to_string(dbg_layer));
    }
    auto down = linear_forward(prod, block.down_proj_lora, block.down_proj_weight);
    if (dbg_layer >= 0 && need_dump_layer(dbg_layer)) {
        dump_tensor(down, "down_proj_out_l" + std::to_string(dbg_layer));
    }
    return down;
}

TensorPtr GemmaModel::forward_block(const TensorPtr& input,
                                    GemmaBlockWeights& block,
                                    const TensorPtr& position_cos,
                                    const TensorPtr& position_sin,
                                    const TensorPtr& pad_mask,
                                    const TensorPtr& base_mask,
                                    float rope_theta) const {
  auto hidden_states = rms_norm(input, block.input_layernorm_weight, config_.rms_norm_eps);
    auto attn_out = apply_attention(hidden_states, block, position_cos, position_sin, pad_mask, base_mask, rope_theta);
    // HF semantics: Gemma3RMSNorm uses (1 + weight) scaling
    attn_out = rms_norm(attn_out, block.post_attention_layernorm_weight, config_.rms_norm_eps);
    hidden_states = add(input, attn_out);

    auto residual = hidden_states;
  hidden_states = rms_norm(hidden_states, block.pre_feedforward_layernorm_weight, config_.rms_norm_eps);
    auto mlp_out = apply_mlp(hidden_states, block);
    // HF semantics: Gemma3RMSNorm uses (1 + weight) scaling
    mlp_out = rms_norm(mlp_out, block.post_feedforward_layernorm_weight, config_.rms_norm_eps);
    hidden_states = add(residual, mlp_out);
    return hidden_states;
}

TensorPtr GemmaModel::forward(const TensorPtr& input_ids,
                              const TensorPtr& attention_mask) {
    auto hidden_states = embedding_lookup(input_ids);
    if (debug_.enabled) {
        dump_tensor(hidden_states, "hidden_states_emb");
    }

    int64_t batch = hidden_states->shape()[0];
    int64_t seq_len = hidden_states->shape()[1];
    int64_t hidden_dim = hidden_states->shape()[2];
    maybe_dump_embedding(hidden_states, batch, seq_len, hidden_dim);

    auto causal_mask = build_causal_mask(seq_len);
    auto sliding_mask = build_sliding_mask(seq_len);
    auto pad_mask = build_padding_mask(attention_mask);

    auto rotary_global = build_rotary_embeddings(batch, seq_len, config_.rope_theta);
    auto rotary_local = build_rotary_embeddings(batch, seq_len, config_.rope_local_base_freq);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        bool is_sliding = i < static_cast<int>(config_.layer_types.size()) &&
                          config_.layer_types[i] == "sliding_attention";
        const auto& pos = is_sliding ? rotary_local : rotary_global;
        float rope_theta = is_sliding ? config_.rope_local_base_freq : config_.rope_theta;
        const auto& mask = is_sliding ? sliding_mask : causal_mask;
        if (!debug_.enabled) {
            hidden_states = forward_block(hidden_states, blocks_[i], pos.cos, pos.sin, pad_mask, mask, rope_theta);
            continue;
        }

        auto normed = rms_norm(hidden_states, blocks_[i].input_layernorm_weight, config_.rms_norm_eps);

        if (need_dump_layer(i)) {
            // Dump the exact tensor fed into attention (input_layernorm output)
            dump_tensor(normed, "hidden_before_attn_l" + std::to_string(i));
            // Also dump selected layer weights for cross-check (MLP weights)
            dump_tensor(blocks_[i].gate_proj_weight, "weights/gate_proj_weight_l" + std::to_string(i));
            dump_tensor(blocks_[i].up_proj_weight, "weights/up_proj_weight_l" + std::to_string(i));
            dump_tensor(blocks_[i].down_proj_weight, "weights/down_proj_weight_l" + std::to_string(i));
        }

        auto attn_out = apply_attention(normed, blocks_[i], pos.cos, pos.sin, pad_mask, mask, rope_theta, i);
        if (need_dump_layer(i)) dump_tensor(attn_out, "hidden_after_attn_l" + std::to_string(i));

        // HF semantics: Gemma3RMSNorm multiplies by (1 + weight)
        attn_out = rms_norm(attn_out, blocks_[i].post_attention_layernorm_weight, config_.rms_norm_eps);
        if (need_dump_layer(i)) dump_tensor(attn_out, "hidden_after_attn_norm_l" + std::to_string(i));
        hidden_states = add(hidden_states, attn_out);
        if (need_dump_layer(i)) dump_tensor(hidden_states, "hidden_after_attn_add_l" + std::to_string(i));

        auto residual = hidden_states;
        auto mlp_in = rms_norm(hidden_states, blocks_[i].pre_feedforward_layernorm_weight, config_.rms_norm_eps);
        if (need_dump_layer(i)) dump_tensor(mlp_in, "hidden_before_mlp_norm_l" + std::to_string(i));
        auto mlp_out = apply_mlp(mlp_in, blocks_[i], i);
        if (need_dump_layer(i)) dump_tensor(mlp_out, "hidden_after_mlp_l" + std::to_string(i));
        // HF semantics: Gemma3RMSNorm multiplies by (1 + weight)
        mlp_out = rms_norm(mlp_out, blocks_[i].post_feedforward_layernorm_weight, config_.rms_norm_eps);
        if (need_dump_layer(i)) dump_tensor(mlp_out, "hidden_after_mlp_norm_l" + std::to_string(i));
        hidden_states = add(residual, mlp_out);
    }

    hidden_states = rms_norm(hidden_states, norm_weight_, config_.rms_norm_eps);
    auto logits = matmul(hidden_states, lm_head_weight_);
    if (debug_.enabled) {
        dump_tensor(logits, "logits");
    }
    return logits;
}

void GemmaModel::assign_weight(const std::string& key, const TensorPtr& tensor) {
    if (key == "embed_tokens.weight") {
        embed_weight_ = tensor;
        if (!lm_head_initialized_) {
            lm_head_weight_ = transpose(tensor, 0, 1);
            lm_head_initialized_ = true;
        }
        return;
    }
    if (key == "norm.weight") {
        norm_weight_ = tensor;
        return;
    }
    if (key == "lm_head.weight") {
        lm_head_weight_ = tensor;
        lm_head_initialized_ = true;
        return;
    }

    std::regex block_pattern(R"(layers\.(\d+)\.(.+))");
    std::smatch match;
    if (!std::regex_match(key, match, block_pattern)) {
        throw std::runtime_error("Unknown Gemma weight key: " + key);
    }

    int layer = std::stoi(match[1].str());
    if (layer < 0 || layer >= config_.num_hidden_layers) {
        throw std::runtime_error("Invalid Gemma layer index: " + std::to_string(layer));
    }
    std::string name = match[2].str();
    auto& block = blocks_[layer];

    if (name == "input_layernorm.weight") block.input_layernorm_weight = tensor;
    else if (name == "post_attention_layernorm.weight") block.post_attention_layernorm_weight = tensor;
    else if (name == "pre_feedforward_layernorm.weight") block.pre_feedforward_layernorm_weight = tensor;
    else if (name == "post_feedforward_layernorm.weight") block.post_feedforward_layernorm_weight = tensor;
    else if (name == "self_attn.q_proj.weight") block.q_proj_weight = tensor;
    else if (name == "self_attn.k_proj.weight") block.k_proj_weight = tensor;
    else if (name == "self_attn.v_proj.weight") block.v_proj_weight = tensor;
    else if (name == "self_attn.o_proj.weight") block.o_proj_weight = tensor;
    else if (name == "self_attn.q_norm.weight") block.q_norm_weight = tensor;
    else if (name == "self_attn.k_norm.weight") block.k_norm_weight = tensor;
    else if (name == "mlp.gate_proj.weight") block.gate_proj_weight = tensor;
    else if (name == "mlp.up_proj.weight") block.up_proj_weight = tensor;
    else if (name == "mlp.down_proj.weight") block.down_proj_weight = tensor;
    else
        throw std::runtime_error("Unknown Gemma block weight: " + name);
}

GemmaBlockWeights& GemmaModel::get_block(int i) {
    if (i < 0 || i >= static_cast<int>(blocks_.size())) {
        throw std::out_of_range("Gemma block index");
    }
    return blocks_[i];
}

const GemmaBlockWeights& GemmaModel::get_block(int i) const {
    if (i < 0 || i >= static_cast<int>(blocks_.size())) {
        throw std::out_of_range("Gemma block index");
    }
    return blocks_[i];
}

void GemmaModel::enable_debug_dump(const std::string& dir, const std::vector<int>& layers) {
    debug_.enabled = true;
    debug_.dir = dir;
    debug_.layers.clear();
    debug_.tensors.clear();
    for (int l : layers) debug_.layers.insert(l);
}

void GemmaModel::disable_debug_dump() {
    debug_ = DebugConfig{};
}

void GemmaModel::dump_layer_norm_weights(int layer, const std::string& dir) const {
    if (layer < 0 || layer >= static_cast<int>(blocks_.size())) return;
    const auto& block = blocks_[layer];
    auto dump_vec = [&](const TensorPtr& w, const std::string& name) {
        if (!w) return;
        if (w->dtype() == DType::kFloat32) {
            save_npy(dir + "/" + name + ".npy", w->data<float>(), w->shape(), DumpDType::kFloat32);
        }
    };
    dump_vec(block.input_layernorm_weight, "weights/input_layernorm_l" + std::to_string(layer));
    dump_vec(block.post_attention_layernorm_weight, "weights/post_attention_layernorm_l" + std::to_string(layer));
    dump_vec(block.pre_feedforward_layernorm_weight, "weights/pre_feedforward_layernorm_l" + std::to_string(layer));
    dump_vec(block.post_feedforward_layernorm_weight, "weights/post_feedforward_layernorm_l" + std::to_string(layer));
    // Extra dump: q_norm/k_norm weights for comparison with PyTorch
    dump_vec(block.q_norm_weight, "weights/q_norm_l" + std::to_string(layer));
    dump_vec(block.k_norm_weight, "weights/k_norm_l" + std::to_string(layer));
}
void GemmaModel::set_numeric_perturb(bool enable, const std::string& name, int64_t index, float eps) {
    debug_.numeric_enabled = enable;
    debug_.numeric_name = name;
    debug_.numeric_index = index;
    debug_.numeric_eps = eps;
}

bool GemmaModel::need_dump_layer(int idx) const {
    return debug_.layers.empty() || debug_.layers.count(idx) > 0;
}

void GemmaModel::dump_tensor(const TensorPtr& t, const std::string& name) const {
    if (!t || !debug_.enabled) return;
    if (t->requires_grad() && debug_.retain_grads) {
        t->retain_grad();  // Retain gradients for non-leaf tensors
    }
    // Cache tensors so gradients can be read after backward
    debug_.tensors[name] = t;
    std::vector<int64_t> shape = t->shape();
    std::string path = debug_.dir + "/" + name + ".npy";
    if (t->dtype() == DType::kFloat32) {
        save_npy(path, t->data<float>(), shape, DumpDType::kFloat32);
    } else if (t->dtype() == DType::kInt32) {
        save_npy(path, t->data<int32_t>(), shape, DumpDType::kInt32);
    }
}

void GemmaModel::init_lora_modules() {
    for (auto& block : blocks_) {
        if (block.lora_initialized) continue;

        block.q_proj_lora = std::make_unique<LoRALinear>(block.q_proj_weight, nullptr);
        block.k_proj_lora = std::make_unique<LoRALinear>(block.k_proj_weight, nullptr);
        block.v_proj_lora = std::make_unique<LoRALinear>(block.v_proj_weight, nullptr);
        block.o_proj_lora = std::make_unique<LoRALinear>(block.o_proj_weight, nullptr);

        block.gate_proj_lora = std::make_unique<LoRALinear>(block.gate_proj_weight, nullptr);
        block.up_proj_lora = std::make_unique<LoRALinear>(block.up_proj_weight, nullptr);
        block.down_proj_lora = std::make_unique<LoRALinear>(block.down_proj_weight, nullptr);

        block.lora_initialized = true;
    }
}

std::vector<TensorPtr> GemmaModel::get_lora_parameters() const {
    std::vector<TensorPtr> params;
    for (const auto& block : blocks_) {
        auto collect = [&](const std::unique_ptr<LoRALinear>& linear) {
            if (!linear) return;
            auto slices = linear->trainable_parameters();
            params.insert(params.end(), slices.begin(), slices.end());
        };
        collect(block.q_proj_lora);
        collect(block.k_proj_lora);
        collect(block.v_proj_lora);
        collect(block.o_proj_lora);
        collect(block.gate_proj_lora);
        collect(block.up_proj_lora);
        collect(block.down_proj_lora);
    }
    return params;
}

void GemmaModel::merge_lora() {
    for (auto& block : blocks_) {
        if (!block.lora_initialized) continue;
        if (block.q_proj_lora) block.q_proj_lora->merge_to_base();
        if (block.k_proj_lora) block.k_proj_lora->merge_to_base();
        if (block.v_proj_lora) block.v_proj_lora->merge_to_base();
        if (block.o_proj_lora) block.o_proj_lora->merge_to_base();
        if (block.gate_proj_lora) block.gate_proj_lora->merge_to_base();
        if (block.up_proj_lora) block.up_proj_lora->merge_to_base();
        if (block.down_proj_lora) block.down_proj_lora->merge_to_base();
    }
}

void GemmaModel::unmerge_lora() {
    for (auto& block : blocks_) {
        if (!block.lora_initialized) continue;
        if (block.q_proj_lora) block.q_proj_lora->unmerge_from_base();
        if (block.k_proj_lora) block.k_proj_lora->unmerge_from_base();
        if (block.v_proj_lora) block.v_proj_lora->unmerge_from_base();
        if (block.o_proj_lora) block.o_proj_lora->unmerge_from_base();
        if (block.gate_proj_lora) block.gate_proj_lora->unmerge_from_base();
        if (block.up_proj_lora) block.up_proj_lora->unmerge_from_base();
        if (block.down_proj_lora) block.down_proj_lora->unmerge_from_base();
    }
}

void GemmaModel::request_embedding_dump(int step, const std::string& output_dir) {
    dump_request_.active = true;
    dump_request_.target_step = step;
    dump_request_.output_dir = output_dir;
    dump_request_.fulfilled = false;
}

void GemmaModel::maybe_dump_embedding(const TensorPtr& hidden_states,
                                      int batch,
                                      int seq_len,
                                      int hidden_dim) {
    if (!dump_request_.active || dump_request_.fulfilled) {
        return;
    }

    const float* data = hidden_states->data<float>();
    int64_t total = static_cast<int64_t>(batch) * seq_len * hidden_dim;
    if (total == 0) {
        dump_request_.fulfilled = true;
        dump_request_.active = false;
        return;
    }

    double sum = 0.0;
    double sumsq = 0.0;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int64_t i = 0; i < total; ++i) {
        float v = data[i];
        sum += v;
        sumsq += static_cast<double>(v) * v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    double mean = sum / static_cast<double>(total);
    double var = sumsq / static_cast<double>(total) - mean * mean;
    if (var < 0.0) var = 0.0;
    double stddev = std::sqrt(var);

    std::cout << "[EmbeddingDump] step " << dump_request_.target_step
              << " shape=[" << batch << "," << seq_len << "," << hidden_dim << "] "
              << std::fixed << std::setprecision(6)
              << "mean=" << mean << " std=" << stddev
              << " min=" << min_val << " max=" << max_val << std::endl;

    int tokens_to_show = std::min<int>(seq_len, 4);
    int dims_to_show = std::min<int>(hidden_dim, 8);
    for (int t = 0; t < tokens_to_show; ++t) {
        std::cout << "  hidden[0," << t << ",0:" << dims_to_show << "] = [";
        for (int d = 0; d < dims_to_show; ++d) {
            int64_t idx = ((0 * seq_len) + t) * hidden_dim + d;
            std::cout << std::setprecision(4) << data[idx];
            if (d + 1 < dims_to_show) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    try {
        std::filesystem::create_directories(dump_request_.output_dir);
        std::string filename = dump_request_.output_dir + "/embedding_step" +
                               std::to_string(dump_request_.target_step) + ".bin";
        std::ofstream out(filename, std::ios::binary);
        if (out) {
            out.write(reinterpret_cast<const char*>(data), total * sizeof(float));
            std::cout << "  [EmbeddingDump] wrote raw tensor to " << filename << std::endl;
        } else {
            std::cerr << "  [EmbeddingDump] failed to write " << filename << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "  [EmbeddingDump] filesystem error: " << e.what() << std::endl;
    }

    dump_request_.fulfilled = true;
    dump_request_.active = false;
}

}  // namespace ops
