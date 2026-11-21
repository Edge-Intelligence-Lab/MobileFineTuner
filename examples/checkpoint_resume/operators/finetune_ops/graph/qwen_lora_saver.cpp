#include "qwen_lora_saver.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "safetensors_loader.h"

namespace ops {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

LoRALinear* resolve_linear(QwenBlock& block, const std::string& target) {
    if (target == "attn.q") return block.q_lin.get();
    if (target == "attn.k") return block.k_lin.get();
    if (target == "attn.v") return block.v_lin.get();
    if (target == "attn.proj") return block.o_lin.get();
    return nullptr;
}

void clear_all_lora(QwenModel& model) {
    for (int i = 0; i < model.config().num_hidden_layers; ++i) {
        auto& blk = model.get_block(i);
        if (blk.q_lin) blk.q_lin->clear_lora();
        if (blk.k_lin) blk.k_lin->clear_lora();
        if (blk.v_lin) blk.v_lin->clear_lora();
        if (blk.o_lin) blk.o_lin->clear_lora();
        blk.lora_initialized = true;
    }
}

}  // namespace

void QwenLoraSaver::save_safetensors(const std::string& path,
                                     const QwenModel& model,
                                     const QwenLoraMetadata& meta) {
    std::unordered_map<std::string, TensorPtr> state;

    auto add_slice = [&](int layer, const std::string& target, const LoRALinear* lin) {
        if (!lin) return;
        const auto& slices = lin->slices();
        if (slices.empty()) return;
        const auto& slice = slices.front();
        if (!slice.A || !slice.B) return;
        const std::string base = "layer." + std::to_string(layer) + "." + target;
        state[base + ".lora_A"] = slice.A;
        state[base + ".lora_B"] = slice.B;
    };

    const auto& cfg = model.config();
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto& blk = model.get_block(i);
        add_slice(i, "attn.q", blk.q_lin.get());
        add_slice(i, "attn.k", blk.k_lin.get());
        add_slice(i, "attn.v", blk.v_lin.get());
        add_slice(i, "attn.proj", blk.o_lin.get());
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open " + path + " for writing");
    }

    std::vector<std::string> keys;
    keys.reserve(state.size());
    for (const auto& kv : state) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    size_t offset = 0;
    std::ostringstream header;
    header << "{";
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        if (idx > 0) header << ",";
        const auto& name = keys[idx];
        const auto& t = state.at(name);
        const auto& shape = t->shape();
        const size_t nbytes = static_cast<size_t>(t->numel()) * sizeof(float);
        header << "\"" << name << "\":{";
        header << "\"dtype\":\"F32\",";
        header << "\"shape\":[";
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d > 0) header << ",";
            header << shape[d];
        }
        header << "],";
        header << "\"data_offsets\":[" << offset << "," << (offset + nbytes) << "]";
        header << "}";
        offset += nbytes;
    }

    header << ",\"__metadata__\":{";
    header << "\"rank\":\"" << meta.rank << "\",";
    header << "\"alpha\":\"" << meta.alpha << "\",";
    header << "\"dropout\":\"" << meta.dropout << "\",";
    header << "\"target_mode\":\"" << (meta.qv_only ? "qv" : "full") << "\"";
    header << "}}";

    const std::string header_str = header.str();
    const uint64_t header_len = static_cast<uint64_t>(header_str.size());
    out.write(reinterpret_cast<const char*>(&header_len), 8);
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));

    for (const auto& name : keys) {
        const auto& t = state.at(name);
        const float* data = t->data<float>();
        const size_t nbytes = static_cast<size_t>(t->numel()) * sizeof(float);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
    }
}

bool QwenLoraSaver::parse_metadata(const std::string& path, QwenLoraMetadata& meta) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 8);
    if (!in) return false;

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in) return false;

    std::smatch match;
    bool found = false;

    const std::regex rank_re(R"(\"rank\"\s*:\s*\"?([0-9]+)\"?)");
    if (std::regex_search(header, match, rank_re) && match.size() > 1) {
        meta.rank = std::stoi(match[1].str());
        found = true;
    }

    const std::regex alpha_re(R"(\"alpha\"\s*:\s*\"?([0-9]+(?:\.[0-9]+)?)\"?)");
    if (std::regex_search(header, match, alpha_re) && match.size() > 1) {
        meta.alpha = std::stof(match[1].str());
        found = true;
    }

    const std::regex dropout_re(R"(\"dropout\"\s*:\s*\"?([0-9]+(?:\.[0-9]+)?)\"?)");
    if (std::regex_search(header, match, dropout_re) && match.size() > 1) {
        meta.dropout = std::stof(match[1].str());
        found = true;
    }

    const std::regex mode_re(R"(\"target_mode\"\s*:\s*\"(qv|full)\")");
    if (std::regex_search(header, match, mode_re) && match.size() > 1) {
        meta.qv_only = (match[1].str() == "qv");
        found = true;
    }

    return found;
}

void QwenLoraSaver::load_safetensors(const std::string& path,
                                     QwenModel& model,
                                     const QwenLoraMetadata& meta) {
    SafeTensorsReader reader(path);
    reader.parse_header();

    clear_all_lora(model);

    struct SliceData {
        TensorPtr A;
        TensorPtr B;
        int layer = -1;
        std::string target;
    };

    std::unordered_map<std::string, SliceData> slices;
    for (const auto& name : reader.get_tensor_names()) {
        if (name == "__metadata__") continue;
        const bool is_a = ends_with(name, ".lora_A");
        const bool is_b = ends_with(name, ".lora_B");
        if (!is_a && !is_b) continue;

        const std::string base = name.substr(0, name.size() - 7);
        auto parts = split(base, '.');
        if (parts.size() != 4 || parts[0] != "layer") {
            continue;
        }

        const int layer = std::stoi(parts[1]);
        const std::string target = parts[2] + "." + parts[3];
        auto& slice = slices[base];
        slice.layer = layer;
        slice.target = target;
        if (is_a) {
            slice.A = reader.load_tensor(name, false);
        } else {
            slice.B = reader.load_tensor(name, false);
        }
    }

    const float scale = meta.rank > 0 ? (meta.alpha / static_cast<float>(meta.rank)) : 1.0f;
    for (const auto& kv : slices) {
        const auto& slice = kv.second;
        if (!slice.A || !slice.B) {
            throw std::runtime_error("Incomplete LoRA slice in " + path + ": " + kv.first);
        }

        auto& blk = model.get_block(slice.layer);
        LoRALinear* lin = resolve_linear(blk, slice.target);
        if (!lin) {
            throw std::runtime_error("Qwen LoRA target not initialized for " + slice.target);
        }
        lin->clear_lora();
        lin->attach_lora(slice.A, slice.B, scale);
    }
}

}  // namespace ops
