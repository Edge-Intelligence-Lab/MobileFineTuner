#include "gemma_lora_injector.h"

#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <regex>

#include "safetensors_loader.h"

namespace ops {

namespace {

struct LinearRef {
    LoRALinear* linear = nullptr;
    TensorPtr weight;
};

LinearRef resolve_linear(GemmaBlockWeights& block, const std::string& name) {
    if (name == "q_proj") return {block.q_proj_lora.get(), block.q_proj_weight};
    if (name == "k_proj") return {block.k_proj_lora.get(), block.k_proj_weight};
    if (name == "v_proj") return {block.v_proj_lora.get(), block.v_proj_weight};
    if (name == "o_proj") return {block.o_proj_lora.get(), block.o_proj_weight};
    if (name == "gate_proj") return {block.gate_proj_lora.get(), block.gate_proj_weight};
    if (name == "up_proj") return {block.up_proj_lora.get(), block.up_proj_weight};
    if (name == "down_proj") return {block.down_proj_lora.get(), block.down_proj_weight};
    return {nullptr, nullptr};
}

std::pair<TensorPtr, TensorPtr> create_lora_params(int64_t in_dim, int64_t out_dim, int rank) {
    // Following PEFT default init: lora_A uses kaiming_uniform_(a=sqrt(5)), lora_B is all zeros.
    // PyTorch computes bound = sqrt(3) * gain / sqrt(fan_in), gain = sqrt(2/(1+a^2)), a=sqrt(5) -> bound = 1/sqrt(fan_in)
    static std::mt19937 rng(42);
    float bound = 1.0f / std::sqrt(static_cast<float>(in_dim));
    std::uniform_real_distribution<float> dist(-bound, bound);

    // Note: In PEFT, lora_A shape is [r, in], lora_B is [out, r]; LoRALinear internally transposes to [in,r]/[r,out]
    auto A = zeros({rank, in_dim}, kFloat32, kCPU);
    float* a_data = A->data<float>();
    for (int64_t i = 0; i < A->numel(); ++i) {
        a_data[i] = dist(rng);
    }

    auto B = zeros({out_dim, rank}, kFloat32, kCPU);
    return {A, B};
}

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

bool parse_lora_metadata(const std::string& path, int& rank_out, float& alpha_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 8);
    if (!in) return false;
    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) return false;

    bool found = false;
    std::smatch match;
    std::regex rank_re(R"(\"rank\"\s*:\s*\"?([0-9]+)\"?)");
    if (std::regex_search(header, match, rank_re) && match.size() > 1) {
        rank_out = std::stoi(match[1].str());
        found = true;
    }
    std::regex alpha_re(R"(\"alpha\"\s*:\s*\"?([0-9]+(?:\.[0-9]+)?)\"?)");
    if (std::regex_search(header, match, alpha_re) && match.size() > 1) {
        alpha_out = std::stof(match[1].str());
        found = true;
    }
    return found;
}

}  // namespace

void GemmaLoraInjector::inject(GemmaModel& model, const GemmaLoraSpec& spec) {
    model_ = &model;
    spec_ = spec;
    num_layers_ = model.config().num_hidden_layers;
    model.init_lora_modules();
    attached_modules_ = 0;

    std::vector<int> layers;
    if (spec.layers.empty()) {
        for (int i = 0; i < num_layers_; ++i) layers.push_back(i);
    } else {
        layers = spec.layers;
    }

    float scale = spec.alpha / static_cast<float>(spec.rank);

    for (int layer : layers) {
        auto& block = model.get_block(layer);
        for (const auto& module : spec.target_modules) {
            auto ref = resolve_linear(block, module);
            if (!ref.linear || !ref.weight) {
                std::cerr << "[GemmaLoraInjector] Skip module " << module
                          << " in layer " << layer << std::endl;
                continue;
            }

            auto shape = ref.weight->shape();
            if (shape.size() != 2) {
                std::cerr << "[GemmaLoraInjector] Unexpected weight shape for " << module << std::endl;
                continue;
            }
            int64_t hidden = model.config().hidden_size;
            int64_t dim0 = shape[0];
            int64_t dim1 = shape[1];
            int64_t in_dim = dim1;
            int64_t out_dim = dim0;

            // Infer in/out direction based on module type/hidden_size
            auto uses_hidden_as_input = (module == "q_proj" || module == "k_proj" || module == "v_proj" ||
                                         module == "gate_proj" || module == "up_proj");
            auto uses_hidden_as_output = (module == "o_proj" || module == "down_proj");

            if (uses_hidden_as_input) {
                if (dim0 == hidden) { in_dim = dim0; out_dim = dim1; }
                else if (dim1 == hidden) { in_dim = dim1; out_dim = dim0; }
            } else if (uses_hidden_as_output) {
                if (dim0 == hidden) { out_dim = dim0; in_dim = dim1; }
                else if (dim1 == hidden) { out_dim = dim1; in_dim = dim0; }
            }

            auto [A, B] = create_lora_params(in_dim, out_dim, spec.rank);
            ref.linear->attach_lora(A, B, scale, 0, out_dim);
            ref.linear->set_debug_name("layers_" + std::to_string(layer) + "_self_attn_" + module);
            attached_modules_++;
        }
    }

    std::cout << "[GemmaLoraInjector] Injected " << attached_modules_
              << " LoRA modules across " << layers.size() << " layers." << std::endl;
}

std::vector<TensorPtr> GemmaLoraInjector::get_trainable_params() const {
    if (!model_) return {};
    return model_->get_lora_parameters();
}

void GemmaLoraInjector::print_info() const {
    std::cout << "[GemmaLoRA]" << std::endl;
    std::cout << "  Rank: " << spec_.rank << std::endl;
    std::cout << "  Alpha: " << spec_.alpha << std::endl;
    std::cout << "  Dropout: " << spec_.dropout << std::endl;
    std::cout << "  Targets: ";
    for (size_t i = 0; i < spec_.target_modules.size(); ++i) {
        std::cout << spec_.target_modules[i];
        if (i + 1 < spec_.target_modules.size()) std::cout << ", ";
    }
    std::cout << std::endl;
}

void GemmaLoraInjector::save_lora_safetensors(const std::string& path) const {
    if (!model_) {
        std::cerr << "[GemmaLoraInjector] save_lora_safetensors: model not set" << std::endl;
        return;
    }
    // Collect LoRA A/B (key names follow PEFT style: layer.{i}.attn.{q|k|v|proj}.lora_{A|B})
    std::unordered_map<std::string, TensorPtr> state;
    auto add_slice = [&](int layer, const std::string& target, const LoRALinear* lin) {
        if (!lin) return;
        const auto& slices = lin->slices();
        if (slices.empty()) return;
        // This implementation expects one slice per module; if multiple slices, write them out with numbering
        for (size_t si = 0; si < slices.size(); ++si) {
            const auto& sl = slices[si];
            if (!sl.A || !sl.B) continue;
            std::string base = "layer." + std::to_string(layer) + "." + target;
            if (slices.size() > 1) base += ("." + std::to_string(si));
            state[base + ".lora_A"] = sl.A;  // A: [r, in] or [in, r] handled internally by LoRALinear, keep as-is here (PEFT style is [r,in])
            state[base + ".lora_B"] = sl.B;  // B: [out, r] or [r, out] same as above
        }
    };
    const auto& cfg = model_->config();
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const auto& blk = model_->get_block(i);
        add_slice(i, "attn.q", blk.q_proj_lora.get());
        add_slice(i, "attn.k", blk.k_proj_lora.get());
        add_slice(i, "attn.v", blk.v_proj_lora.get());
        add_slice(i, "attn.proj", blk.o_proj_lora.get());
        // If MLP LoRA is enabled in the future, can add mappings below (not exported now to avoid conflicts with other implementations):
        // add_slice(i, "mlp.gate", blk.gate_proj_lora.get());
        // add_slice(i, "mlp.up",   blk.up_proj_lora.get());
        // add_slice(i, "mlp.down", blk.down_proj_lora.get());
    }
    // Output safetensors
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[GemmaLoraInjector] Cannot open " << path << " for writing" << std::endl;
        return;
    }
    // Sort keys for stability
    std::vector<std::string> keys;
    keys.reserve(state.size());
    for (const auto& kv : state) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    // Build header JSON
    size_t offset = 0;
    std::ostringstream header;
    header << "{";
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        if (idx > 0) header << ",";
        const auto& name = keys[idx];
        const auto& t = state.at(name);
        const auto& shape = t->shape();
        size_t nbytes = static_cast<size_t>(t->numel()) * sizeof(float);
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
    // Metadata
    header << ",\"__metadata__\":{";
    header << "\"rank\":\"" << spec_.rank << "\",";
    header << "\"alpha\":\"" << spec_.alpha << "\",";
    header << "\"dropout\":\"" << spec_.dropout << "\",";
    header << "\"targets\":\"attn.q,attn.k,attn.v,attn.proj\"";
    header << "}}";
    std::string header_str = header.str();
    uint64_t header_len = static_cast<uint64_t>(header_str.size());
    out.write(reinterpret_cast<const char*>(&header_len), 8);
    out.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));
    // Write tensor data consecutively (F32)
    for (const auto& name : keys) {
        const auto& t = state.at(name);
        const float* data = t->data<float>();
        size_t nbytes = static_cast<size_t>(t->numel()) * sizeof(float);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
    }
    out.close();
    std::cout << "[GemmaLoraInjector] Saved " << keys.size()
              << " LoRA tensors to " << path << std::endl;
}

void GemmaLoraInjector::load_lora_safetensors(const std::string& path) {
    if (!model_) {
        std::cerr << "[GemmaLoraInjector] load_lora_safetensors: model not set" << std::endl;
        return;
    }

    model_->init_lora_modules();
    for (int i = 0; i < model_->config().num_hidden_layers; ++i) {
        auto& blk = model_->get_block(i);
        auto clear = [](const std::unique_ptr<LoRALinear>& lin) {
            if (lin) lin->clear_lora();
        };
        clear(blk.q_proj_lora);
        clear(blk.k_proj_lora);
        clear(blk.v_proj_lora);
        clear(blk.o_proj_lora);
        clear(blk.gate_proj_lora);
        clear(blk.up_proj_lora);
        clear(blk.down_proj_lora);
    }

    int meta_rank = -1;
    float meta_alpha = -1.0f;
    if (parse_lora_metadata(path, meta_rank, meta_alpha)) {
        if (meta_rank > 0) spec_.rank = meta_rank;
        if (meta_alpha > 0.0f) spec_.alpha = meta_alpha;
    }
    float scale = (spec_.rank > 0) ? (spec_.alpha / static_cast<float>(spec_.rank)) : 1.0f;
    if (spec_.rank <= 0) {
        std::cerr << "[GemmaLoraInjector] Warning: invalid rank, using scale=1.0" << std::endl;
    }

    SafeTensorsReader reader(path);
    reader.parse_header();
    auto names = reader.get_tensor_names();

    struct SliceData {
        TensorPtr A;
        TensorPtr B;
        int layer = -1;
        std::string target;
        int idx = 0;
    };
    std::unordered_map<std::string, SliceData> slices;

    for (const auto& name : names) {
        if (name == "__metadata__") continue;
        bool is_A = ends_with(name, ".lora_A");
        bool is_B = ends_with(name, ".lora_B");
        if (!is_A && !is_B) continue;

        std::string base = name.substr(0, name.size() - 7); // strip .lora_A/.lora_B
        auto parts = split(base, '.');
        if (parts.size() < 4 || parts[0] != "layer") {
            std::cerr << "[GemmaLoraInjector] Skip unrecognized key: " << name << std::endl;
            continue;
        }
        int layer = std::stoi(parts[1]);
        std::string target = parts[2] + "." + parts[3];
        int slice_idx = 0;
        if (parts.size() >= 5) {
            slice_idx = std::stoi(parts[4]);
        }
        std::string key = std::to_string(layer) + ":" + target + ":" + std::to_string(slice_idx);

        auto t = reader.load_tensor(name, false);
        auto& entry = slices[key];
        entry.layer = layer;
        entry.target = target;
        entry.idx = slice_idx;
        if (is_A) entry.A = t;
        else entry.B = t;
    }

    auto map_target = [](const std::string& t) -> std::string {
        if (t == "attn.q") return "q_proj";
        if (t == "attn.k") return "k_proj";
        if (t == "attn.v") return "v_proj";
        if (t == "attn.proj") return "o_proj";
        if (t == "mlp.gate") return "gate_proj";
        if (t == "mlp.up") return "up_proj";
        if (t == "mlp.down") return "down_proj";
        return "";
    };

    int attached = 0;
    for (auto& kv : slices) {
        auto& s = kv.second;
        if (!s.A || !s.B) {
            std::cerr << "[GemmaLoraInjector] Incomplete LoRA slice: " << kv.first << std::endl;
            continue;
        }
        if (s.layer < 0 || s.layer >= model_->config().num_hidden_layers) {
            std::cerr << "[GemmaLoraInjector] Invalid layer index: " << s.layer << std::endl;
            continue;
        }
        std::string module = map_target(s.target);
        if (module.empty()) {
            std::cerr << "[GemmaLoraInjector] Unsupported target: " << s.target << std::endl;
            continue;
        }
        auto& blk = model_->get_block(s.layer);
        auto ref = resolve_linear(blk, module);
        if (!ref.linear) {
            std::cerr << "[GemmaLoraInjector] Missing module " << module
                      << " in layer " << s.layer << std::endl;
            continue;
        }
        int64_t cols = s.B->shape().empty() ? 0 : s.B->shape()[0];
        if (cols <= 0) {
            std::cerr << "[GemmaLoraInjector] Invalid B shape for " << kv.first << std::endl;
            continue;
        }
        ref.linear->attach_lora(s.A, s.B, scale, 0, static_cast<int>(cols));
        attached++;
    }

    attached_modules_ = attached;
    std::cout << "[GemmaLoraInjector] Loaded " << attached
              << " LoRA slices from " << path << std::endl;
}

void GemmaLoraInjector::merge_all(GemmaModel& model) {
    model.merge_lora();
}

void GemmaLoraInjector::unmerge_all(GemmaModel& model) {
    model.unmerge_lora();
}

}  // namespace ops
