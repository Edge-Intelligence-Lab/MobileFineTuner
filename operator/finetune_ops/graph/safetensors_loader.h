/**
 * @file safetensors_loader.h
 * @brief SafeTensors weight loader (pure C++)
 *
 * Supports:
 * - Parse safetensors format (8B header_len + JSON header + raw data)
 * - Load FP32/FP16 tensors (FP16 auto-promoted to FP32)
 * - Key mapping (HF → internal naming)
 * - Auto-transpose Linear weights ([out,in] → [in,out])
 */

#pragma once

#include "../core/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>

namespace ops {

/**
 * @brief Tensor metadata in a SafeTensors file
 */
struct SafeTensorInfo {
    std::string dtype;              // "F32", "F16", "I32", "I64", etc.
    std::vector<int64_t> shape;     // tensor shape
    std::vector<size_t> data_offsets;  // [start, end) in file
};

/**
 * @brief SafeTensors load options
 */
struct SafeTensorsLoadOptions {
    bool transpose_linear = true;   // auto-transpose Linear weights [out,in]→[in,out]
    bool auto_promote_fp16 = true;  // auto-promote FP16 to FP32
    bool verbose = true;            // print load logs
    bool strict_shape_check = true; // strict shape validation
    // Preserve original F16/BF16 storage for matching internal or HF keys even
    // when auto_promote_fp16 is true. Used for frozen base weights where
    // trainable grads and optimizer state remain FP32.
    std::vector<std::string> preserve_low_precision_key_substrings;
};

/**
 * @brief SafeTensors file reader
 */
class SafeTensorsReader {
public:
    explicit SafeTensorsReader(const std::string& filepath);
    ~SafeTensorsReader();
    
    /**
     * @brief Parse file header (8B header_len + JSON header)
     */
    void parse_header();
    
    /**
     * @brief List all tensor keys
     */
    std::vector<std::string> get_tensor_names() const;
    
    /**
     * @brief Get metadata for a tensor
     */
    SafeTensorInfo get_tensor_info(const std::string& name) const;
    
    /**
     * @brief Load a tensor into memory
     * @param name tensor name
     * @param transpose whether to transpose (for 2D tensors)
     * @return tensor pointer
     */
    TensorPtr load_tensor(const std::string& name, bool transpose = false);
    
    /**
     * @brief Load tensors with key mapping
     * @param key_mapping {"internal_key": "hf_key"}
     * @param options load options
     * @return {"internal_key": tensor}
     */
    std::unordered_map<std::string, TensorPtr> 
    load_tensors_mapped(const std::unordered_map<std::string, std::string>& key_mapping,
                        const SafeTensorsLoadOptions& options = SafeTensorsLoadOptions());

private:
    std::string filepath_;
    std::ifstream file_;
    size_t header_len_;
    size_t data_offset_;  // data start offset after header
    std::unordered_map<std::string, SafeTensorInfo> tensor_map_;
    
    void parse_tensor_metadata(const std::string& json_str);
    TensorPtr read_tensor_data(const SafeTensorInfo& info, bool transpose, bool preserve_low_precision);
};

/**
 * @brief SafeTensors model reader for either a single model.safetensors file or
 * a HuggingFace sharded snapshot with model.safetensors.index.json.
 */
class SafeTensorsModelReader {
public:
    explicit SafeTensorsModelReader(const std::string& model_dir_or_file);

    void parse_headers();

    std::vector<std::string> get_tensor_names() const;

    std::unordered_map<std::string, TensorPtr>
    load_tensors_mapped(const std::unordered_map<std::string, std::string>& key_mapping,
                        const SafeTensorsLoadOptions& options = SafeTensorsLoadOptions());

    const std::vector<std::string>& files() const { return files_; }

    static std::vector<std::string> resolve_weight_files(const std::string& model_dir_or_file);

private:
    std::string model_dir_or_file_;
    std::vector<std::string> files_;
    std::vector<std::unique_ptr<SafeTensorsReader>> readers_;
    std::unordered_map<std::string, size_t> tensor_to_reader_;
};

/**
 * @brief GPT-2 HuggingFace → internal key mapping generator
 */
class GPT2KeyMapper {
public:
    /**
     * @brief Generate mapping (GPT-2 12 layers, n_embd=768)
     * @param num_layers number of layers (default 12)
     * @return {"internal_key": "hf_key"}
     */
    static std::unordered_map<std::string, std::string> 
    generate_gpt2_mapping(int num_layers = 12);
    
    /**
     * @brief Print mapping (debug)
     */
    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

/**
 * @brief Gemma3 HuggingFace → internal key mapping generator
 */
class GemmaKeyMapper {
public:
    static std::unordered_map<std::string, std::string>
    generate_gemma_mapping(int num_layers = 18);

    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

/**
 * @brief Qwen2.5 HuggingFace → internal key mapping generator
 */
class QwenKeyMapper {
public:
    /**
     * @brief Generate Qwen2.5 mapping
     * @param num_layers number of layers (e.g., 24)
     * @return {"internal_key": "hf_key"}
     */
    static std::unordered_map<std::string, std::string>
    generate_qwen_mapping(int num_layers = 24);
};

}  // namespace ops
