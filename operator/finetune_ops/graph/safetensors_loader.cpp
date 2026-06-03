/**
 * @file safetensors_loader.cpp
 * @brief SafeTensors format weight loader implementation
 */

#include "safetensors_loader.h"
#include "../core/ops.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <regex>
#include <cstring>
#include <filesystem>
#include <set>
#include <algorithm>

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
    // 1. Read header_len (first 8 bytes, little-endian uint64)
    uint64_t header_len_raw;
    file_.read(reinterpret_cast<char*>(&header_len_raw), 8);
    if (!file_) {
        throw std::runtime_error("Failed to read header_len");
    }
    header_len_ = static_cast<size_t>(header_len_raw);
    
    // 2. Read JSON header
    std::vector<char> header_bytes(header_len_);
    file_.read(header_bytes.data(), header_len_);
    if (!file_) {
        throw std::runtime_error("Failed to read JSON header");
    }
    
    std::string header_json(header_bytes.begin(), header_bytes.end());
    data_offset_ = 8 + header_len_;
    
    // 3. Lightweight JSON parsing (manually extract tensor metadata)
    parse_tensor_metadata(header_json);
}

void SafeTensorsReader::parse_tensor_metadata(const std::string& json_str) {
    // Lightweight regex extraction: find every "tensor_name": {...}
    std::regex tensor_pattern(R"#("([^"]+)"\s*:\s*\{[^}]+\})#");
    auto tensor_begin = std::sregex_iterator(json_str.begin(), json_str.end(), tensor_pattern);
    auto tensor_end = std::sregex_iterator();
    
    for (auto it = tensor_begin; it != tensor_end; ++it) {
        std::string name = (*it)[1].str();
        std::string block = (*it)[0].str();
        
        // Extract dtype
        std::regex dtype_pattern(R"#("dtype"\s*:\s*"([^"]+)")#");
        std::smatch dtype_match;
        std::string dtype = "F32";
        if (std::regex_search(block, dtype_match, dtype_pattern)) {
            dtype = dtype_match[1].str();
        }
        
        // Extract shape
        std::regex shape_pattern(R"#("shape"\s*:\s*\[([^\]]+)\])#");
        std::smatch shape_match;
        std::vector<int64_t> shape;
        if (std::regex_search(block, shape_match, shape_pattern)) {
            std::string shape_str = shape_match[1].str();
            std::istringstream iss(shape_str);
            std::string val;
            while (std::getline(iss, val, ',')) {
                // Trim whitespace
                val.erase(std::remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) {
                    shape.push_back(std::stoll(val));
                }
            }
        }
        
        // Extract data_offsets
        std::regex offsets_pattern(R"#("data_offsets"\s*:\s*\[(\d+)\s*,\s*(\d+)\])#");
        std::smatch offsets_match;
        std::vector<size_t> offsets;
        if (std::regex_search(block, offsets_match, offsets_pattern)) {
            offsets.push_back(std::stoull(offsets_match[1].str()));
            offsets.push_back(std::stoull(offsets_match[2].str()));
        }
        
        SafeTensorInfo info;
        info.dtype = dtype;
        info.shape = shape;
        info.data_offsets = offsets;
        
        tensor_map_[name] = info;
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
    auto fp16_to_fp32_inline = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;
        uint32_t f32_bits;
        if (exponent == 0) {
            f32_bits = (mantissa == 0) ? (sign << 31) : 0;
        } else if (exponent == 0x1F) {
            f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f32_bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &f32_bits, sizeof(float));
        return result;
    };

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

    if (byte_size != static_cast<size_t>(numel) * element_size) {
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
    auto transpose_buffer_float = [&](float* buffer) {
        if (!need_transpose) return;
        int64_t rows = info.shape[0];
        int64_t cols = info.shape[1];
        std::vector<float> temp(numel);
        std::memcpy(temp.data(), buffer, numel * sizeof(float));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                buffer[j * rows + i] = temp[i * cols + j];
            }
        }
    };
    auto transpose_buffer_u16 = [&](uint16_t* buffer) {
        if (!need_transpose) return;
        int64_t rows = info.shape[0];
        int64_t cols = info.shape[1];
        std::vector<uint16_t> temp(static_cast<size_t>(numel));
        std::memcpy(temp.data(), buffer, static_cast<size_t>(numel) * sizeof(uint16_t));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                buffer[j * rows + i] = temp[i * cols + j];
            }
        }
    };
    auto transpose_buffer_i32 = [&](int32_t* buffer) {
        if (!need_transpose) return;
        int64_t rows = info.shape[0];
        int64_t cols = info.shape[1];
        std::vector<int32_t> temp(static_cast<size_t>(numel));
        std::memcpy(temp.data(), buffer, static_cast<size_t>(numel) * sizeof(int32_t));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                buffer[j * rows + i] = temp[i * cols + j];
            }
        }
    };

    if (info.dtype == "F32") {
        std::memcpy(tensor->data<float>(), raw_data.data(), byte_size);
        transpose_buffer_float(tensor->data<float>());
    } else if (info.dtype == "F16") {
        const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw_data.data());
        if (target_dtype == kFloat16) {
            std::memcpy(tensor->data<uint16_t>(), fp16_data, byte_size);
            transpose_buffer_u16(tensor->data<uint16_t>());
        } else {
            float* fp32_data = tensor->data<float>();
            for (int64_t i = 0; i < numel; ++i) {
                fp32_data[i] = fp16_to_fp32_inline(fp16_data[i]);
            }
            transpose_buffer_float(fp32_data);
        }
    } else if (info.dtype == "BF16") {
        const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(raw_data.data());
        if (target_dtype == kBFloat16) {
            std::memcpy(tensor->data<uint16_t>(), bf16_data, byte_size);
            transpose_buffer_u16(tensor->data<uint16_t>());
        } else {
            float* fp32_data = tensor->data<float>();
            for (int64_t i = 0; i < numel; ++i) {
                uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
                std::memcpy(&fp32_data[i], &bits, sizeof(float));
            }
            transpose_buffer_float(fp32_data);
        }
    } else if (info.dtype == "I32") {
        std::memcpy(tensor->data<int32_t>(), raw_data.data(), byte_size);
        transpose_buffer_i32(tensor->data<int32_t>());
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
        bool transpose = options.transpose_linear && 
                        (hf_key.find("weight") != std::string::npos) &&
                        (hf_key.find("ln") == std::string::npos) &&
                        !is_embedding &&
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
        std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::regex shard_pattern(R"#("([^"]+)"\s*:\s*"([^"]+\.safetensors)")#");
        std::set<std::string> unique_names;
        auto begin = std::sregex_iterator(json.begin(), json.end(), shard_pattern);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            unique_names.insert((*it)[2].str());
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

}  // namespace ops
