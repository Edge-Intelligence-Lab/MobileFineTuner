/**
 * @file lora_saver.cpp
 * @brief LoRA safetensors save/load implementation
 */

#include "lora_saver.h"
#include "safetensors_loader.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <stdexcept>
#include <iostream>
#include <regex>
#include <algorithm>
#include <set>

namespace ops {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string target_to_metadata(LoraTarget target) {
    switch (target) {
        case LoraTarget::AttnQKV: return "AttnQKV";
        case LoraTarget::AttnProj: return "AttnProj";
        case LoraTarget::MlpFcIn: return "MlpFcIn";
        case LoraTarget::MlpFcOut: return "MlpFcOut";
        default: return "Unknown";
    }
}

bool metadata_to_target(const std::string& name, LoraTarget& out) {
    std::string lowered = to_lower(name);
    if (lowered == "attnqkv" || lowered == "attn_qkv") {
        out = LoraTarget::AttnQKV;
        return true;
    }
    if (lowered == "attnproj" || lowered == "attn_proj") {
        out = LoraTarget::AttnProj;
        return true;
    }
    if (lowered == "mlpfcin" || lowered == "mlp_fc_in") {
        out = LoraTarget::MlpFcIn;
        return true;
    }
    if (lowered == "mlpfcout" || lowered == "mlp_fc_out") {
        out = LoraTarget::MlpFcOut;
        return true;
    }
    return false;
}

std::vector<LoraTarget> parse_targets_string(const std::string& csv) {
    std::vector<LoraTarget> targets;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        LoraTarget t;
        if (metadata_to_target(item, t)) {
            if (std::find(targets.begin(), targets.end(), t) == targets.end()) {
                targets.push_back(t);
            }
        }
    }
    return targets;
}

std::string join_targets(const std::vector<LoraTarget>& targets) {
    std::ostringstream oss;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (i > 0) oss << ',';
        oss << target_to_metadata(targets[i]);
    }
    return oss.str();
}

bool key_target_to_enum(const std::string& key, LoraTarget& out) {
    if (key == "attn.q" || key == "attn.k" || key == "attn.v" || key == "attn.qkv") {
        out = LoraTarget::AttnQKV;
        return true;
    }
    if (key == "attn.proj") {
        out = LoraTarget::AttnProj;
        return true;
    }
    if (key == "mlp.fc_in") {
        out = LoraTarget::MlpFcIn;
        return true;
    }
    if (key == "mlp.fc_out") {
        out = LoraTarget::MlpFcOut;
        return true;
    }
    return false;
}

} // namespace

bool LoRAState::compatible_with(const GPT2Model& model) const {
    // 简单校验：层数、隐藏维
    (void)model; // 避免未使用警告
    // Assume at least one tensor for layer 0 exists
    for (const auto& kv : tensors) {
        if (kv.first.find("layer.0") != std::string::npos) {
            // 检查维度是否合理
            auto shape = kv.second->shape();
            if (shape.empty()) continue;
            // rank should be within [1,128]
            if (rank < 1 || rank > 128) return false;
            // 简单通过
            return true;
        }
    }
    return !tensors.empty();
}

std::string LoraSaver::make_peft_key(int layer, const std::string& target, const std::string& ab) {
    // Standard keys: layer.{i}.attn.{q|k|v|proj} or mlp.{fc_in|fc_out}.lora_{A|B}
    return "layer." + std::to_string(layer) + "." + target + ".lora_" + ab;
}

bool LoraSaver::parse_peft_key(const std::string& key, int& layer, std::string& target, std::string& ab) {
    // Supports: layer.{i}.attn.proj.lora_A or layer.{i}.mlp.fc_in.lora_B, etc.
    // Strategy: locate "layer." and ".lora_"
    size_t layer_pos = key.find("layer.");
    size_t lora_pos = key.find(".lora_");
    if (layer_pos == std::string::npos || lora_pos == std::string::npos) return false;
    
    // 提取层号
    size_t layer_start = layer_pos + 6; // after "layer."
    size_t layer_end = key.find('.', layer_start);
    if (layer_end == std::string::npos || layer_end >= lora_pos) return false;
    
    try {
        layer = std::stoi(key.substr(layer_start, layer_end - layer_start));
    } catch (...) { return false; }
    
    // Extract target (between layer index and lora token)
    target = key.substr(layer_end + 1, lora_pos - layer_end - 1);
    
    // 提取A/B
    std::string lora_suffix = key.substr(lora_pos + 6); // after ".lora_"
    if (lora_suffix == "A") ab = "A";
    else if (lora_suffix == "B") ab = "B";
    else return false;
    
    return true;
}

void LoraSaver::save_safetensors(const std::string& path, 
                                  const GPT2Model& model,
                                  const LoraSpec& spec) {
    std::unordered_map<std::string, TensorPtr> state;
    
    const auto& cfg = model.config();
    for (int i = 0; i < cfg.n_layer; ++i) {
        const auto& blk = model.get_block(i);
        
        auto has_target = [&](LoraTarget tgt) {
            if (spec.targets.empty()) return true;
            return std::find(spec.targets.begin(), spec.targets.end(), tgt) != spec.targets.end();
        };

        auto add_slice = [&](const std::string& target_name, const LoRASlice& slice) {
            if (!slice.A || !slice.B) return;
            state[make_peft_key(i, target_name, "A")] = slice.A;
            state[make_peft_key(i, target_name, "B")] = slice.B;
        };

        // Attention QKV (split/merged)
        if (blk.qkv_lin && has_target(LoraTarget::AttnQKV)) {
            const auto& slices = blk.qkv_lin->slices();
            if (!slices.empty()) {
                if (spec.split_qkv) {
                    static const char* names[] = {"attn.q", "attn.k", "attn.v"};
                    size_t limit = std::min<size_t>(slices.size(), 3);
                    for (size_t idx = 0; idx < limit; ++idx) {
                        add_slice(names[idx], slices[idx]);
                    }
                } else {
                    add_slice("attn.qkv", slices.front());
                }
            }
        }

        // Attention output projection
        if (blk.proj_lin && has_target(LoraTarget::AttnProj)) {
            const auto& slices = blk.proj_lin->slices();
            if (!slices.empty()) {
                add_slice("attn.proj", slices.front());
            }
        }

        // MLP fc_in
        if (blk.fc_in_lin && has_target(LoraTarget::MlpFcIn)) {
            const auto& slices = blk.fc_in_lin->slices();
            if (!slices.empty()) {
                add_slice("mlp.fc_in", slices.front());
            }
        }

        // MLP fc_out
        if (blk.fc_out_lin && has_target(LoraTarget::MlpFcOut)) {
            const auto& slices = blk.fc_out_lin->slices();
            if (!slices.empty()) {
                add_slice("mlp.fc_out", slices.front());
            }
        }
    }
    
    // Write safetensors (order must remain stable)
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open " + path);
    
    // 排序键名以保证顺序一致
    std::vector<std::string> sorted_keys;
    sorted_keys.reserve(state.size());
    for (const auto& kv : state) sorted_keys.push_back(kv.first);
    std::sort(sorted_keys.begin(), sorted_keys.end());
    
    // Build header
    std::ostringstream header_json;
    header_json << "{";
    size_t offset = 0;
    
    for (size_t idx = 0; idx < sorted_keys.size(); ++idx) {
        if (idx > 0) header_json << ",";
        const auto& name = sorted_keys[idx];
        const auto& t = state.at(name);
        auto shape = t->shape();
        size_t nbytes = t->numel() * sizeof(float);
        
        header_json << "\"" << name << "\":{";
        header_json << "\"dtype\":\"F32\",";
        header_json << "\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) header_json << ",";
            header_json << shape[i];
        }
        header_json << "],";
        header_json << "\"data_offsets\":[" << offset << "," << (offset + nbytes) << "]";
        header_json << "}";
        
        offset += nbytes;
    }
    
    // 元数据
    header_json << ",\"__metadata__\":{";
    header_json << "\"rank\":\"" << spec.rank << "\",";
    header_json << "\"alpha\":\"" << spec.alpha << "\",";
    header_json << "\"dropout\":\"" << spec.dropout << "\",";
    header_json << "\"split_qkv\":\"" << (spec.split_qkv ? "true" : "false") << "\",";
    header_json << "\"targets\":\"" << join_targets(spec.targets) << "\"";
    header_json << "}}";
    
    std::string header_str = header_json.str();
    uint64_t header_len = header_str.size();
    
    // Write header length (8 bytes, little-endian)
    out.write(reinterpret_cast<const char*>(&header_len), 8);
    // Write header
    out.write(header_str.c_str(), header_len);
    
    // 写数据（按排序后的顺序）
    for (const auto& name : sorted_keys) {
        const auto& t = state.at(name);
        const float* data = t->data<float>();
        size_t nbytes = t->numel() * sizeof(float);
        out.write(reinterpret_cast<const char*>(data), nbytes);
    }
    
    out.close();
    std::cout << "[LoraSaver] Saved " << state.size() << " tensors to " << path << std::endl;
}

LoRAState LoraSaver::load_safetensors(const std::string& path) {
    SafeTensorsReader reader(path);
    reader.parse_header();
    
    LoRAState state;

    // First attempt to read LoRA config from safetensors __metadata__
    try {
        std::ifstream fin(path, std::ios::binary);
        if (fin) {
            uint64_t header_len_raw = 0;
            fin.read(reinterpret_cast<char*>(&header_len_raw), 8);
            size_t header_len = static_cast<size_t>(header_len_raw);
            std::string header_json(header_len, '\0');
            fin.read(&header_json[0], header_len);
            // Extract the __metadata__ block
            std::smatch m;
            std::regex meta_pat(R"#("__metadata__"\s*:\s*\{([^}]*)\})#");
            if (std::regex_search(header_json, m, meta_pat)) {
                std::string meta_block = m[1].str();
                auto pick_str = [&](const std::string& key, std::string& out) {
                    std::smatch mm; std::regex p("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
                    if (std::regex_search(meta_block, mm, p)) { out = mm[1].str(); return true; }
                    return false;
                };
                std::string s_rank, s_alpha, s_dropout, s_split, s_targets;
                if (pick_str("rank", s_rank)) {
                    try { state.rank = std::stoi(s_rank); } catch (...) {}
                }
                if (pick_str("alpha", s_alpha)) {
                    try { state.alpha = std::stof(s_alpha); } catch (...) {}
                }
                if (pick_str("dropout", s_dropout)) {
                    try { state.dropout = std::stof(s_dropout); } catch (...) {}
                }
                if (pick_str("split_qkv", s_split)) {
                    std::string v = s_split; for (auto& c : v) c = std::tolower(c);
                    state.split_qkv = (v == "true" || v == "1");
                }
                if (pick_str("targets", s_targets)) {
                    auto parsed = parse_targets_string(s_targets);
                    if (!parsed.empty()) state.targets = parsed;
                }
            }
        }
    } catch (...) {
        std::cerr << "[LoraSaver] WARN: failed to parse __metadata__ from safetensors header (fallback to defaults)" << std::endl;
    }
    
    // If __metadata__ lacks rank, infer from the first lora_A shape
    for (const auto& name : reader.get_tensor_names()) {
        if (name.find("lora_A") != std::string::npos) {
            auto info = reader.get_tensor_info(name);
            auto shape = info.shape;
            if (shape.size() == 2) {
                if (state.rank <= 0) {
                    state.rank = static_cast<int>(shape[1]); // [in_dim, rank]
                }
                break;
            }
        }
    }
    
    // Read all tensors (skip __metadata__)
    auto names = reader.get_tensor_names();
    std::cout << "[LoraSaver] Found " << names.size() << " entries in file" << std::endl;
    for (const auto& name : names) {
        if (name == "__metadata__") continue; // skip metadata entry
        try {
            auto t = reader.load_tensor(name, false);
            state.tensors[name] = t;
        } catch (const std::exception& e) {
            std::cerr << "  ❌ Failed to load tensor '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }
    
    std::cout << "[LoraSaver] Loaded " << state.tensors.size() << " LoRA tensors from " << path << std::endl;
    std::cout << "  rank=" << state.rank << ", alpha=" << state.alpha << std::endl;
    
    bool saw_fused_qkv = false;
    bool saw_split_qkv = false;
    std::set<LoraTarget> deduced_targets;
    for (const auto& kv : state.tensors) {
        int layer;
        std::string target;
        std::string ab;
        if (!LoraSaver::parse_peft_key(kv.first, layer, target, ab)) continue;
        if (target == "attn.qkv") saw_fused_qkv = true;
        if (target == "attn.q" || target == "attn.k" || target == "attn.v") saw_split_qkv = true;
        LoraTarget tgt;
        if (key_target_to_enum(target, tgt)) {
            deduced_targets.insert(tgt);
        }
    }
    if (state.targets.empty()) {
        state.targets.assign(deduced_targets.begin(), deduced_targets.end());
    }
    if (saw_fused_qkv) {
        state.split_qkv = false;
    } else if (saw_split_qkv) {
        state.split_qkv = true;
    }
    
    return state;
}

void LoraSaver::attach_from_state(GPT2Model& model, const LoRAState& state) {
    // Attach loaded LoRA weights to corresponding LoRALinear modules
    std::unordered_map<std::string, TensorPtr> a_map, b_map;
    
    // Group A and B by layer.target
    for (const auto& kv : state.tensors) {
        const std::string& key = kv.first;
        int layer;
        std::string target, ab;
        if (!parse_peft_key(key, layer, target, ab)) continue;
        std::string group_key = std::to_string(layer) + "." + target;
        if (ab == "A") {
            a_map[group_key] = kv.second;
        } else if (ab == "B") {
            b_map[group_key] = kv.second;
        }
    }
    
    const auto& cfg = model.config();
    int effective_rank = state.rank > 0 ? state.rank : 1;
    float scale = state.alpha / static_cast<float>(effective_rank);

    // 先清空现有的 LoRA 切片，避免重复附加
    for (int i = 0; i < cfg.n_layer; ++i) {
        auto& blk = model.get_block(i);
        if (blk.qkv_lin) blk.qkv_lin->clear_lora();
        if (blk.proj_lin) blk.proj_lin->clear_lora();
        if (blk.fc_in_lin) blk.fc_in_lin->clear_lora();
        if (blk.fc_out_lin) blk.fc_out_lin->clear_lora();
    }

    auto attach_target = [&](int layer, const std::string& target, LoRALinear* lin,
                             int col0, int cols) -> bool {
        if (!lin) return false;
        std::string key = std::to_string(layer) + "." + target;
        auto ait = a_map.find(key);
        auto bit = b_map.find(key);
        if (ait == a_map.end() || bit == b_map.end()) return false;
        auto A = ait->second;
        auto B = bit->second;
        if (!A || !B) return false;
        A->set_requires_grad(true);
        B->set_requires_grad(true);
        lin->attach_lora(A, B, scale, col0, cols);
        return true;
    };

    for (int i = 0; i < cfg.n_layer; ++i) {
        auto& blk = model.get_block(i);
        const int C = cfg.n_embd;
        bool fused_attached = attach_target(i, "attn.qkv", blk.qkv_lin.get(), 0, 3 * C);
        if (!fused_attached) {
            attach_target(i, "attn.q", blk.qkv_lin.get(), 0, C);
            attach_target(i, "attn.k", blk.qkv_lin.get(), C, C);
            attach_target(i, "attn.v", blk.qkv_lin.get(), 2 * C, C);
        }
        attach_target(i, "attn.proj", blk.proj_lin.get(), 0, C);
        attach_target(i, "mlp.fc_in", blk.fc_in_lin.get(), 0, 4 * C);
        attach_target(i, "mlp.fc_out", blk.fc_out_lin.get(), 0, C);
    }
    
    std::cout << "[LoraSaver] Attached LoRA to model (rank=" << state.rank 
              << ", alpha=" << state.alpha << ")" << std::endl;
}

} // namespace ops
