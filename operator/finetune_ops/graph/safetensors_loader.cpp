/**
 * @file safetensors_loader.cpp
 * @brief SafeTensors format weight loader implementation
 */

#include "safetensors_loader.h"
#include "../../third_party/nlohmann/json.hpp"
#include "../core/ops.h"
#include "qwen_model.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ops {

// ============================================================================
// SafeTensorsReader implementation
// ============================================================================

SafeTensorsReader::SafeTensorsReader(const std::string& filepath)
    : filepath_(filepath), header_len_(0), data_offset_(0) {
    file_.open(filepath, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open safetensors file: " + filepath);
    }
}

SafeTensorsReader::~SafeTensorsReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

void SafeTensorsReader::parse_header() {
    file_.clear();
    file_.seekg(0, std::ios::end);
    const auto length = file_.tellg();
    if (length < 8)
        throw std::runtime_error("Truncated safetensors header");
    file_.seekg(0, std::ios::beg);
    uint64_t header_len_raw = 0;
    file_.read(reinterpret_cast<char*>(&header_len_raw), 8);
    if (!file_ || header_len_raw > 100000000 || header_len_raw > static_cast<uint64_t>(length) - 8)
        throw std::runtime_error("Invalid safetensors header length");
    header_len_ = static_cast<size_t>(header_len_raw);
    data_offset_ = 8 + header_len_;
    std::string header(header_len_, ' ');
    file_.read(header.data(), header.size());
    if (!file_)
        throw std::runtime_error("Failed to read safetensors header");
    tensor_map_.clear();
    parse_tensor_metadata(header);
    const uint64_t data_bytes = static_cast<uint64_t>(length) - data_offset_;
    for (const auto& entry : tensor_map_)
        if (entry.second.data_offsets[1] > data_bytes)
            throw std::runtime_error("Safetensors payload truncated: " + entry.first);
}

void SafeTensorsReader::parse_tensor_metadata(const std::string& text) {
    const auto header = nlohmann::json::parse(text);
    if (!header.is_object())
        throw std::runtime_error("Safetensors header must be a JSON object");
    for (const auto& item : header.items()) {
        if (item.key() == "__metadata__")
            continue;
        SafeTensorInfo info;
        info.dtype = item.value().at("dtype").get<std::string>();
        const auto& shape = item.value().at("shape");
        if (!shape.is_array())
            throw std::runtime_error("Invalid safetensors shape");
        for (const auto& dim : shape)
            if (!dim.is_number_integer() || dim < 0 || dim > INT64_MAX)
                throw std::runtime_error("Safetensors dimensions must be nonnegative integers");
        info.shape = shape.get<std::vector<int64_t>>();
        const auto& offsets = item.value().at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2 || !offsets[0].is_number_unsigned() ||
            !offsets[1].is_number_unsigned())
            throw std::runtime_error("Invalid safetensors data offsets");
        info.data_offsets = offsets.get<std::vector<size_t>>();
        if (info.data_offsets[1] < info.data_offsets[0])
            throw std::runtime_error("Reversed safetensors data offsets");
        uint64_t numel = 1;
        for (auto dim : info.shape) {
            if (dim < 0 || (dim > 0 && numel > static_cast<uint64_t>(INT64_MAX) / dim))
                throw std::runtime_error("Invalid safetensors shape");
            numel *= dim;
        }
        tensor_map_.emplace(item.key(), std::move(info));
    }
}

std::vector<std::string> SafeTensorsReader::get_tensor_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensor_map_) {
        names.push_back(name);
    }
    return names;
}

SafeTensorInfo SafeTensorsReader::get_tensor_info(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

TensorPtr SafeTensorsReader::load_tensor(const std::string& name, bool transpose) {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    return read_tensor_data(it->second, transpose, false);
}

TensorPtr SafeTensorsReader::read_tensor_data(const SafeTensorInfo& info, bool transpose, bool preserve_low_precision) {
    if (info.data_offsets.size() != 2) {
        throw std::runtime_error("Invalid data_offsets");
    }

    size_t start = info.data_offsets[0];
    size_t end = info.data_offsets[1];
    size_t byte_size = end - start;

    int64_t numel = 1;
    for (auto dim : info.shape) {
        numel *= dim;
    }

    size_t element_size = 4;
    DType target_dtype = kFloat32;

    if (info.dtype == "F32") {
        element_size = 4;
        target_dtype = kFloat32;
    } else if (info.dtype == "F16") {
        element_size = 2;
        target_dtype = preserve_low_precision ? kFloat16 : kFloat32;
    } else if (info.dtype == "BF16") {
        element_size = 2;
        target_dtype = preserve_low_precision ? kBFloat16 : kFloat32;
    } else if (info.dtype == "I32") {
        element_size = 4;
        target_dtype = kInt32;
    } else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

    if (static_cast<uint64_t>(numel) > std::numeric_limits<size_t>::max() / element_size ||
        byte_size != static_cast<size_t>(numel) * element_size) {
        throw std::runtime_error("Size mismatch for tensor");
    }

    file_.seekg(data_offset_ + start, std::ios::beg);
    std::vector<char> raw_data(byte_size);
    file_.read(raw_data.data(), byte_size);
    if (!file_) {
        throw std::runtime_error("Failed to read tensor data");
    }

    std::vector<int64_t> shape = info.shape;
    bool need_transpose = transpose && shape.size() == 2;
    if (need_transpose) {
        std::swap(shape[0], shape[1]);
    }

    TensorPtr tensor = std::make_shared<Tensor>(shape, target_dtype, kCPU);
    // Convert and transpose directly from the file buffer into the final tensor.
    // Keeping a second full FP32 transpose buffer can otherwise dominate loading
    // peak memory for a large vocabulary projection.
    auto copy_converted = [&](const auto* source, auto* destination, auto convert) {
        if (!need_transpose) {
            for (int64_t i = 0; i < numel; ++i)
                destination[i] = convert(source[i]);
            return;
        }
        const int64_t rows = info.shape[0], cols = info.shape[1];
        constexpr int64_t tile = 32;
        for (int64_t row = 0; row < rows; row += tile)
            for (int64_t col = 0; col < cols; col += tile)
                for (int64_t i = row; i < std::min(row + tile, rows); ++i)
                    for (int64_t j = col; j < std::min(col + tile, cols); ++j)
                        destination[j * rows + i] = convert(source[i * cols + j]);
    };
    if (info.dtype == "F32") {
        copy_converted(reinterpret_cast<const float*>(raw_data.data()), tensor->data<float>(),
                       [](float value) { return value; });
    } else if (info.dtype == "I32") {
        copy_converted(reinterpret_cast<const int32_t*>(raw_data.data()), tensor->data<int32_t>(),
                       [](int32_t value) { return value; });
    } else {
        const auto* source = reinterpret_cast<const uint16_t*>(raw_data.data());
        if (target_dtype == kFloat16 || target_dtype == kBFloat16) {
            copy_converted(source, tensor->data<uint16_t>(), [](uint16_t value) { return value; });
        } else if (info.dtype == "F16") {
            copy_converted(source, tensor->data<float>(), fp16_bits_to_float32);
        } else {
            copy_converted(source, tensor->data<float>(), [](uint16_t value) {
                const uint32_t bits = static_cast<uint32_t>(value) << 16;
                float converted;
                std::memcpy(&converted, &bits, sizeof(float));
                return converted;
            });
        }
    }

    return tensor;
}

std::unordered_map<std::string, TensorPtr> 
SafeTensorsReader::load_tensors_mapped(
    const std::unordered_map<std::string, std::string>& key_mapping,
    const SafeTensorsLoadOptions& options) {
    
    std::unordered_map<std::string, TensorPtr> result;
    
    for (const auto& [internal_key, hf_key] : key_mapping) {
        auto it = tensor_map_.find(hf_key);
        if (it == tensor_map_.end()) {
            if (options.verbose) {
                std::cerr << "[WARN] HF key not found: " << hf_key << std::endl;
            }
            continue;
        }
        
        // Decide whether transpose is needed (linear layer weights)
        bool is_embedding = (hf_key.find("wte") != std::string::npos) ||
                            (hf_key.find("wpe") != std::string::npos) ||
                            (hf_key.find("embed_tokens") != std::string::npos);
        bool transpose =
            options.transpose_linear &&
            std::find(options.skip_transpose_keys.begin(), options.skip_transpose_keys.end(),
                      internal_key) == options.skip_transpose_keys.end() &&
            (hf_key.find("weight") != std::string::npos) &&
            (hf_key.find("ln") == std::string::npos) && !is_embedding &&
            it->second.shape.size() == 2;

        bool preserve_low_precision = !options.auto_promote_fp16;
        if (options.auto_promote_fp16) {
            for (const auto& needle : options.preserve_low_precision_key_substrings) {
                if ((!needle.empty()) &&
                    (internal_key.find(needle) != std::string::npos ||
                     hf_key.find(needle) != std::string::npos)) {
                    preserve_low_precision = true;
                    break;
                }
            }
        }

        auto tensor = read_tensor_data(it->second, transpose, preserve_low_precision);
        result[internal_key] = tensor;
        
        if (options.verbose) {
            std::cout << "[Loaded] " << internal_key << " <- " << hf_key 
                      << " shape=[";
            for (size_t i = 0; i < tensor->shape().size(); ++i) {
                std::cout << tensor->shape()[i];
                if (i < tensor->shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (transpose) std::cout << " (transposed)";
            std::cout << std::endl;
        }
    }
    
    return result;
}

// ============================================================================
// SafeTensorsModelReader implementation
// ============================================================================

SafeTensorsModelReader::SafeTensorsModelReader(const std::string& model_dir_or_file)
    : model_dir_or_file_(model_dir_or_file) {}

std::vector<std::string> SafeTensorsModelReader::resolve_weight_files(const std::string& model_dir_or_file) {
    namespace fs = std::filesystem;

    fs::path input(model_dir_or_file);
    if (fs::is_regular_file(input)) {
        return {input.string()};
    }
    if (!fs::is_directory(input)) {
        throw std::runtime_error("SafeTensors path is neither file nor directory: " + model_dir_or_file);
    }

    fs::path single = input / "model.safetensors";
    if (fs::is_regular_file(single)) {
        return {single.string()};
    }

    fs::path index = input / "model.safetensors.index.json";
    if (fs::is_regular_file(index)) {
        std::ifstream f(index);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open SafeTensors index: " + index.string());
        }
        const auto index_json = nlohmann::json::parse(f);
        const auto& weight_map = index_json.at("weight_map");
        if (!weight_map.is_object())
            throw std::runtime_error("SafeTensors weight_map must be an object");
        std::set<std::string> unique_names;
        for (const auto& item : weight_map.items()) {
            const auto name = item.value().get<std::string>();
            const fs::path relative(name);
            if (relative.is_absolute() || relative.extension() != ".safetensors")
                throw std::runtime_error("Invalid SafeTensors shard filename: " + name);
            for (const auto& part : relative)
                if (part == "..")
                    throw std::runtime_error(
                        "SafeTensors shard must stay inside the model directory");
            unique_names.insert(name);
        }
        if (unique_names.empty()) {
            throw std::runtime_error("SafeTensors index contains no shard filenames: " + index.string());
        }

        std::vector<std::string> files;
        for (const auto& name : unique_names) {
            fs::path shard = input / name;
            if (!fs::is_regular_file(shard)) {
                throw std::runtime_error("SafeTensors shard listed in index is missing: " + shard.string());
            }
            files.push_back(shard.string());
        }
        return files;
    }

    std::vector<std::string> fallback_shards;
    for (const auto& entry : fs::directory_iterator(input)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            fallback_shards.push_back(entry.path().string());
        }
    }
    std::sort(fallback_shards.begin(), fallback_shards.end());
    if (!fallback_shards.empty()) {
        return fallback_shards;
    }

    throw std::runtime_error(
        "No SafeTensors weights found in " + model_dir_or_file +
        ". Expected model.safetensors or model.safetensors.index.json with shard files.");
}

void SafeTensorsModelReader::parse_headers() {
    files_ = resolve_weight_files(model_dir_or_file_);
    readers_.clear();
    tensor_to_reader_.clear();

    for (const auto& file : files_) {
        auto reader = std::make_unique<SafeTensorsReader>(file);
        reader->parse_header();
        size_t reader_index = readers_.size();
        for (const auto& name : reader->get_tensor_names()) {
            auto inserted = tensor_to_reader_.emplace(name, reader_index);
            if (!inserted.second) {
                throw std::runtime_error("Duplicate tensor key across SafeTensors shards: " + name);
            }
        }
        readers_.push_back(std::move(reader));
    }
}

std::vector<std::string> SafeTensorsModelReader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_reader_.size());
    for (const auto& kv : tensor_to_reader_) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::unordered_map<std::string, TensorPtr>
SafeTensorsModelReader::load_tensors_mapped(
    const std::unordered_map<std::string, std::string>& key_mapping,
    const SafeTensorsLoadOptions& options) {
    if (readers_.empty()) {
        throw std::logic_error("SafeTensorsModelReader::parse_headers must be called before loading tensors");
    }

    std::vector<std::unordered_map<std::string, std::string>> grouped(readers_.size());
    for (const auto& [internal_key, hf_key] : key_mapping) {
        auto it = tensor_to_reader_.find(hf_key);
        if (it == tensor_to_reader_.end()) {
            if (options.verbose) {
                std::cerr << "[WARN] HF key not found in SafeTensors model: " << hf_key << std::endl;
            }
            continue;
        }
        grouped[it->second][internal_key] = hf_key;
    }

    std::unordered_map<std::string, TensorPtr> result;
    for (size_t i = 0; i < readers_.size(); ++i) {
        if (grouped[i].empty()) {
            continue;
        }
        auto partial = readers_[i]->load_tensors_mapped(grouped[i], options);
        result.insert(partial.begin(), partial.end());
    }
    return result;
}

// ============================================================================
// GPT2KeyMapper implementation
// ============================================================================

std::unordered_map<std::string, std::string> 
GPT2KeyMapper::generate_gpt2_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> mapping;
    
    // Embeddings
    mapping["wte.weight"] = "wte.weight";
    mapping["wpe.weight"] = "wpe.weight";
    
    // Transformer blocks
    for (int i = 0; i < num_layers; ++i) {
        std::string hf_prefix = "h." + std::to_string(i) + ".";
        std::string internal_prefix = "blocks." + std::to_string(i) + ".";
        
        // LayerNorm 1
        mapping[internal_prefix + "ln_1.weight"] = hf_prefix + "ln_1.weight";
        mapping[internal_prefix + "ln_1.bias"] = hf_prefix + "ln_1.bias";
        
        // Attention
        mapping[internal_prefix + "attn.qkv.weight"] = hf_prefix + "attn.c_attn.weight";
        mapping[internal_prefix + "attn.qkv.bias"] = hf_prefix + "attn.c_attn.bias";
        mapping[internal_prefix + "attn.proj.weight"] = hf_prefix + "attn.c_proj.weight";
        mapping[internal_prefix + "attn.proj.bias"] = hf_prefix + "attn.c_proj.bias";
        
        // LayerNorm 2
        mapping[internal_prefix + "ln_2.weight"] = hf_prefix + "ln_2.weight";
        mapping[internal_prefix + "ln_2.bias"] = hf_prefix + "ln_2.bias";
        
        // MLP
        mapping[internal_prefix + "mlp.fc_in.weight"] = hf_prefix + "mlp.c_fc.weight";
        mapping[internal_prefix + "mlp.fc_in.bias"] = hf_prefix + "mlp.c_fc.bias";
        mapping[internal_prefix + "mlp.fc_out.weight"] = hf_prefix + "mlp.c_proj.weight";
        mapping[internal_prefix + "mlp.fc_out.bias"] = hf_prefix + "mlp.c_proj.bias";
    }
    
    // Final LayerNorm
    mapping["ln_f.weight"] = "ln_f.weight";
    mapping["ln_f.bias"] = "ln_f.bias";
    
    // lm_head (typically tied with wte; enable below if loading separately)
    // mapping["lm_head.weight"] = "lm_head.weight";
    
    return mapping;
}

void GPT2KeyMapper::print_mapping(const std::unordered_map<std::string, std::string>& mapping) {
    std::cout << "\n[GPT2 Key Mapping] Total: " << mapping.size() << " entries\n";
    for (const auto& [internal, hf] : mapping) {
        std::cout << "  " << internal << " <- " << hf << std::endl;
    }
}

std::unordered_map<std::string, std::string> 
GemmaKeyMapper::generate_gemma_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> mapping;
    mapping["embed_tokens.weight"] = "model.embed_tokens.weight";
    mapping["norm.weight"] = "model.norm.weight";
    mapping["lm_head.weight"] = "lm_head.weight";

    for (int i = 0; i < num_layers; ++i) {
        std::string internal_prefix = "layers." + std::to_string(i);
        std::string hf_prefix = "model.layers." + std::to_string(i);

        mapping[internal_prefix + ".input_layernorm.weight"] = hf_prefix + ".input_layernorm.weight";
        mapping[internal_prefix + ".post_attention_layernorm.weight"] = hf_prefix + ".post_attention_layernorm.weight";
        mapping[internal_prefix + ".pre_feedforward_layernorm.weight"] =
            hf_prefix + ".pre_feedforward_layernorm.weight";
        mapping[internal_prefix + ".post_feedforward_layernorm.weight"] =
            hf_prefix + ".post_feedforward_layernorm.weight";

        mapping[internal_prefix + ".self_attn.q_proj.weight"] = hf_prefix + ".self_attn.q_proj.weight";
        mapping[internal_prefix + ".self_attn.k_proj.weight"] = hf_prefix + ".self_attn.k_proj.weight";
        mapping[internal_prefix + ".self_attn.v_proj.weight"] = hf_prefix + ".self_attn.v_proj.weight";
        mapping[internal_prefix + ".self_attn.o_proj.weight"] = hf_prefix + ".self_attn.o_proj.weight";
        mapping[internal_prefix + ".self_attn.q_norm.weight"] = hf_prefix + ".self_attn.q_norm.weight";
        mapping[internal_prefix + ".self_attn.k_norm.weight"] = hf_prefix + ".self_attn.k_norm.weight";

        mapping[internal_prefix + ".mlp.gate_proj.weight"] = hf_prefix + ".mlp.gate_proj.weight";
        mapping[internal_prefix + ".mlp.up_proj.weight"] = hf_prefix + ".mlp.up_proj.weight";
        mapping[internal_prefix + ".mlp.down_proj.weight"] = hf_prefix + ".mlp.down_proj.weight";
    }

    return mapping;
}

void GemmaKeyMapper::print_mapping(const std::unordered_map<std::string, std::string>& mapping) {
    for (const auto& kv : mapping) {
        std::cout << kv.first << " -> " << kv.second << std::endl;
    }
}

// ============================================================================
// QwenKeyMapper implementation
// ============================================================================

std::unordered_map<std::string, std::string>
QwenKeyMapper::generate_qwen_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> m;
    // Embeddings and final norm
    m["embed_tokens.weight"] = "model.embed_tokens.weight";
    m["final_norm.weight"] = "model.norm.weight";
    // Per-layer mappings
    for (int i = 0; i < num_layers; ++i) {
        std::string hf = "model.layers." + std::to_string(i) + ".";
        std::string in = "layers." + std::to_string(i) + ".";
        m[in + "input_norm.weight"] = hf + "input_layernorm.weight";
        m[in + "post_norm.weight"] = hf + "post_attention_layernorm.weight";
        m[in + "self_attn.q_proj.weight"] = hf + "self_attn.q_proj.weight";
        m[in + "self_attn.q_proj.bias"]   = hf + "self_attn.q_proj.bias";
        m[in + "self_attn.k_proj.weight"] = hf + "self_attn.k_proj.weight";
        m[in + "self_attn.k_proj.bias"]   = hf + "self_attn.k_proj.bias";
        m[in + "self_attn.v_proj.weight"] = hf + "self_attn.v_proj.weight";
        m[in + "self_attn.v_proj.bias"]   = hf + "self_attn.v_proj.bias";
        m[in + "self_attn.o_proj.weight"] = hf + "self_attn.o_proj.weight";
        m[in + "mlp.gate_proj.weight"]    = hf + "mlp.gate_proj.weight";
        m[in + "mlp.up_proj.weight"]      = hf + "mlp.up_proj.weight";
        m[in + "mlp.down_proj.weight"]    = hf + "mlp.down_proj.weight";
    }
    return m;
}

std::unordered_map<std::string, std::string>
QwenKeyMapper::generate_qwen_mapping(const QwenConfig& config) {
    config.validate();
    if (config.hybrid()) {
        std::unordered_map<std::string, std::string> m;
        m["embed_tokens.weight"] = config.checkpoint_prefix + "embed_tokens.weight";
        m["final_norm.weight"] = config.checkpoint_prefix + "norm.weight";
        if (!config.tie_word_embeddings)
            m["lm_head.weight"] = "lm_head.weight";
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            const auto in = "layers." + std::to_string(i) + ".", hf = config.checkpoint_prefix + in;
            m[in + "input_norm.weight"] = hf + "input_layernorm.weight";
            m[in + "post_norm.weight"] = hf + "post_attention_layernorm.weight";
            for (const auto* key : {"gate_proj", "up_proj", "down_proj"})
                m[in + "mlp." + key + ".weight"] = hf + "mlp." + key + ".weight";
            if (config.linear_layer(i)) {
                for (const auto* key : {"in_proj_qkv.weight", "in_proj_z.weight",
                                        "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight",
                                        "conv1d.weight", "A_log", "dt_bias", "norm.weight"})
                    m[in + "linear_attn." + key] = hf + "linear_attn." + key;
            } else {
                for (const auto* key : {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"})
                    m[in + "self_attn." + key + ".weight"] = hf + "self_attn." + key + ".weight";
            }
        }
        return m;
    }
    auto mapping = generate_qwen_mapping(config.num_hidden_layers);
    if (!config.attention_bias) {
        for (auto it = mapping.begin(); it != mapping.end();) {
            if (it->first.size() >= 5 && it->first.compare(it->first.size() - 5, 5, ".bias") == 0)
                it = mapping.erase(it);
            else
                ++it;
        }
    }
    if (config.uses_qk_norm())
        for (int i = 0; i < config.num_hidden_layers; ++i)
            for (const auto* name : {"q_norm", "k_norm"})
                mapping["layers." + std::to_string(i) + ".self_attn." + name + ".weight"] =
                    "model.layers." + std::to_string(i) + ".self_attn." + name + ".weight";
    if (!config.tie_word_embeddings)
        mapping["lm_head.weight"] = "lm_head.weight";
    if (config.model_type == "qwen3" && config.attention_bias)
        for (int i = 0; i < config.num_hidden_layers; ++i)
            mapping["layers." + std::to_string(i) + ".self_attn.o_proj.bias"] =
                "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
    return mapping;
}

}  // namespace ops
