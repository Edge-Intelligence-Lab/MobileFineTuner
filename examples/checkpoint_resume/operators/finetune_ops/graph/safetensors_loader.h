/**
 * @file safetensors_loader.h
 * @brief SafeTensors format weight loader (pure C++ implementation)
 * 
 * Supports:
 * - Parse safetensors format (8B header_len + JSON header + raw data)
 * - Load FP32/FP16 tensors (FP16 auto-promoted to FP32)
 * - Key name mapping (HF -> internal naming)
 * - Auto-transpose Linear weights ([out,in] -> [in,out])
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
 * @brief Tensor metadata in SafeTensors file
 */
struct SafeTensorInfo {
    std::string dtype;              // "F32", "F16", "I32", "I64" etc.
    std::vector<int64_t> shape;     // Tensor shape
    std::vector<size_t> data_offsets;  // [start, end) in file
};

/**
 * @brief SafeTensors load options
 */
struct SafeTensorsLoadOptions {
    bool transpose_linear = true;   // Auto-transpose linear layer weights [out,in]→[in,out]
    bool auto_promote_fp16 = true;  // Auto-promote FP16 to FP32
    bool verbose = true;            // Print loading log
    bool strict_shape_check = true; // Strict shape validation
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
     * @brief Get list of all tensor key names
     */
    std::vector<std::string> get_tensor_names() const;
    
    /**
     * @brief Get metadata for specified tensor
     */
    SafeTensorInfo get_tensor_info(const std::string& name) const;
    
    /**
     * @brief Load specified tensor to memory
     * @param name Tensor name
     * @param transpose Whether to transpose (for 2D tensors)
     * @return Tensor pointer
     */
    TensorPtr load_tensor(const std::string& name, bool transpose = false);
    
    /**
     * @brief Batch load tensors (with key mapping)
     * @param key_mapping {"internal_key": "hf_key"}
     * @param options Load options
     * @return {"internal_key": tensor}
     */
    std::unordered_map<std::string, TensorPtr> 
    load_tensors_mapped(const std::unordered_map<std::string, std::string>& key_mapping,
                        const SafeTensorsLoadOptions& options = SafeTensorsLoadOptions());

private:
    std::string filepath_;
    std::ifstream file_;
    size_t header_len_;
    size_t data_offset_;  // Data start position after header
    std::unordered_map<std::string, SafeTensorInfo> tensor_map_;
    
    void parse_tensor_metadata(const std::string& json_str);
    TensorPtr read_tensor_data(const SafeTensorInfo& info, bool transpose);
};

/**
 * @brief GPT-2 HuggingFace -> internal key name mapping generator
 */
class GPT2KeyMapper {
public:
    /**
     * @brief Generate complete mapping table (GPT-2 12 layers, n_embd=768)
     * @param num_layers Layer count (default 12)
     * @return {"internal_key": "hf_key"}
     */
    static std::unordered_map<std::string, std::string> 
    generate_gpt2_mapping(int num_layers = 12);
    
    /**
     * @brief Print mapping table (for debugging)
     */
    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

/**
 * @brief Gemma3 HuggingFace -> internal key name mapping generator
 */
class GemmaKeyMapper {
public:
    static std::unordered_map<std::string, std::string>
    generate_gemma_mapping(int num_layers = 18);

    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

/**
 * @brief Qwen2.5 HuggingFace -> internal key name mapping generator
 */
class QwenKeyMapper {
public:
    static std::unordered_map<std::string, std::string>
    generate_qwen_mapping(int num_layers = 24);

    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

}  // namespace ops
