/**
 * @file activation_compressor.cpp
 * @brief Mobile-optimized activation compression implementation
 * 
 * Implements INT8/INT4 quantization and sparsification for activation tensors.
 * Optimized for mobile/edge devices with memory constraints.
 */

#include "activation_compressor.h"
#include "../core/ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <chrono>

namespace ops {
namespace memory {

// ============================================================================
// ActivationCompressor Implementation
// ============================================================================

ActivationCompressor::ActivationCompressor(const CompressionConfig& config)
    : config_(config),
      total_compressions_(0),
      total_decompressions_(0),
      total_bytes_saved_(0),
      total_compression_time_(0.0),
      total_decompression_time_(0.0),
      current_memory_pressure_(0.0f),
      current_battery_level_(100),
      is_thermal_throttling_(false) {
    
    initialize_hardware_accelerators();
}

ActivationCompressor::~ActivationCompressor() {
    clear_compressed_cache();
}

void ActivationCompressor::initialize_hardware_accelerators() {
    // Placeholder for hardware acceleration initialization
    // Can add NEON/GPU acceleration in future
}

// ============================================================================
// Main Compression/Decompression API
// ============================================================================

std::unique_ptr<CompressedActivation> ActivationCompressor::compress_activation(
    const TensorPtr& activation,
    ActivationCompressionMode compression_mode,
    size_t activation_id) {
    
    if (!activation) return nullptr;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::unique_ptr<CompressedActivation> compressed;
    
    switch (compression_mode) {
        case ActivationCompressionMode::QUANTIZE_INT8:
            compressed = quantize_int8(activation, activation_id);
            break;
            
        case ActivationCompressionMode::QUANTIZE_INT4:
            compressed = quantize_int4(activation, activation_id);
            break;
            
        case ActivationCompressionMode::SPARSE_50:
            compressed = sparsify_activation(activation, 0.5f, activation_id);
            break;
            
        case ActivationCompressionMode::SPARSE_75:
            compressed = sparsify_activation(activation, 0.75f, activation_id);
            break;
            
        case ActivationCompressionMode::LOSSY_COMPRESS:
            compressed = lossy_compress(activation, config_.lossy_compression_ratio, activation_id);
            break;
            
        case ActivationCompressionMode::ADAPTIVE:
            {
                auto optimal_mode = select_optimal_compression_mode(activation, "", true);
                return compress_activation(activation, optimal_mode, activation_id);
            }
            
        case ActivationCompressionMode::NONE:
        default:
            return nullptr;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (compressed && compressed->metadata) {
        update_compression_statistics(*compressed->metadata, elapsed_ms);
    }
    
    total_compressions_++;
    
    return compressed;
}

TensorPtr ActivationCompressor::decompress_activation(const CompressedActivation& compressed) {
    if (!compressed.metadata) return nullptr;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    TensorPtr decompressed;
    
    switch (compressed.metadata->mode) {
        case ActivationCompressionMode::QUANTIZE_INT8:
            decompressed = dequantize_int8(compressed);
            break;
            
        case ActivationCompressionMode::QUANTIZE_INT4:
            decompressed = dequantize_int4(compressed);
            break;
            
        case ActivationCompressionMode::SPARSE_50:
        case ActivationCompressionMode::SPARSE_75:
            decompressed = desparsify_activation(compressed);
            break;
            
        case ActivationCompressionMode::LOSSY_COMPRESS:
            decompressed = lossy_decompress(compressed);
            break;
            
        default:
            return nullptr;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    double current = total_decompression_time_.load();
    while (!total_decompression_time_.compare_exchange_weak(current, current + elapsed_ms));
    total_decompressions_++;
    
    return decompressed;
}

// ============================================================================
// INT8 Quantization
// ============================================================================

std::unique_ptr<CompressedActivation> ActivationCompressor::quantize_int8(
    const TensorPtr& activation, size_t activation_id) {
    
    if (!activation || activation->dtype() != kFloat32) return nullptr;
    
    const float* data = activation->data<float>();
    size_t numel = static_cast<size_t>(activation->numel());
    
    // Calculate scale and zero_point
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < numel; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    // Compute quantization parameters
    float scale = (max_val - min_val) / 255.0f;
    if (scale < 1e-8f) scale = 1e-8f;  // Avoid division by zero
    int zero_point = static_cast<int>(std::round(-min_val / scale));
    zero_point = std::max(0, std::min(255, zero_point));
    
    // Allocate compressed buffer
    auto compressed = std::make_unique<CompressedActivation>(
        numel * sizeof(int8_t) + 2 * sizeof(float),  // data + scale + zero_point
        ActivationCompressionMode::QUANTIZE_INT8
    );
    
    // Store metadata
    compressed->metadata->scale = scale;
    compressed->metadata->zero_point = zero_point;
    compressed->metadata->original_dtype = activation->dtype();
    compressed->metadata->original_size = numel * sizeof(float);
    compressed->metadata->compressed_size = numel * sizeof(int8_t);
    compressed->metadata->compression_ratio = 
        static_cast<float>(compressed->metadata->original_size) / compressed->metadata->compressed_size;
    compressed->original_shape = activation->shape();
    
    // Quantize: q = clamp(round((x - zero_point) / scale), -128, 127)
    compressed->compressed_data.resize(numel * sizeof(int8_t) + 2 * sizeof(float));
    int8_t* quantized = reinterpret_cast<int8_t*>(compressed->compressed_data.data());
    
    for (size_t i = 0; i < numel; ++i) {
        float val = data[i];
        int q_val = static_cast<int>(std::round(val / scale)) - zero_point;
        q_val = std::max(-128, std::min(127, q_val));
        quantized[i] = static_cast<int8_t>(q_val);
    }
    
    // Store scale and zero_point at the end
    float* metadata_ptr = reinterpret_cast<float*>(quantized + numel);
    metadata_ptr[0] = scale;
    metadata_ptr[1] = static_cast<float>(zero_point);
    
    total_bytes_saved_ += (numel * sizeof(float) - numel * sizeof(int8_t));
    
    return compressed;
}

TensorPtr ActivationCompressor::dequantize_int8(const CompressedActivation& compressed) {
    if (!compressed.metadata || compressed.metadata->mode != ActivationCompressionMode::QUANTIZE_INT8) {
        return nullptr;
    }
    
    size_t numel = compressed.metadata->original_size / sizeof(float);
    const int8_t* quantized = reinterpret_cast<const int8_t*>(compressed.compressed_data.data());
    
    // Retrieve scale and zero_point
    const float* metadata_ptr = reinterpret_cast<const float*>(quantized + numel);
    float scale = metadata_ptr[0];
    int zero_point = static_cast<int>(metadata_ptr[1]);
    
    // Create output tensor
    auto decompressed = ops::zeros(compressed.original_shape, compressed.metadata->original_dtype);
    float* output = decompressed->data<float>();
    
    // Dequantize: x = (q + zero_point) * scale
    for (size_t i = 0; i < numel; ++i) {
        output[i] = (static_cast<int>(quantized[i]) + zero_point) * scale;
    }
    
    return decompressed;
}

// ============================================================================
// INT4 Quantization
// ============================================================================

std::unique_ptr<CompressedActivation> ActivationCompressor::quantize_int4(
    const TensorPtr& activation, size_t activation_id) {
    
    if (!activation || activation->dtype() != kFloat32) return nullptr;
    
    const float* data = activation->data<float>();
    size_t numel = static_cast<size_t>(activation->numel());
    
    // Calculate scale (INT4: -8 to 7)
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < numel; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    float scale = (max_val - min_val) / 15.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    int zero_point = static_cast<int>(std::round(-min_val / scale));
    zero_point = std::max(0, std::min(15, zero_point));
    
    // Pack 2 INT4 values into 1 byte
    size_t packed_size = (numel + 1) / 2;
    auto compressed = std::make_unique<CompressedActivation>(
        packed_size + 2 * sizeof(float),
        ActivationCompressionMode::QUANTIZE_INT4
    );
    
    compressed->metadata->scale = scale;
    compressed->metadata->zero_point = zero_point;
    compressed->metadata->original_dtype = activation->dtype();
    compressed->metadata->original_size = numel * sizeof(float);
    compressed->metadata->compressed_size = packed_size;
    compressed->metadata->compression_ratio = 
        static_cast<float>(compressed->metadata->original_size) / compressed->metadata->compressed_size;
    compressed->original_shape = activation->shape();
    
    compressed->compressed_data.resize(packed_size + 2 * sizeof(float));
    uint8_t* packed = compressed->compressed_data.data();
    
    // Quantize and pack
    for (size_t i = 0; i < numel; i += 2) {
        // First value (lower 4 bits)
        float val1 = data[i];
        int q1 = static_cast<int>(std::round(val1 / scale)) - zero_point;
        q1 = std::max(-8, std::min(7, q1)) + 8;  // Map to 0-15
        
        // Second value (upper 4 bits)
        int q2 = 0;
        if (i + 1 < numel) {
            float val2 = data[i + 1];
            q2 = static_cast<int>(std::round(val2 / scale)) - zero_point;
            q2 = std::max(-8, std::min(7, q2)) + 8;
        }
        
        packed[i / 2] = static_cast<uint8_t>((q2 << 4) | q1);
    }
    
    // Store metadata
    float* metadata_ptr = reinterpret_cast<float*>(packed + packed_size);
    metadata_ptr[0] = scale;
    metadata_ptr[1] = static_cast<float>(zero_point);
    
    total_bytes_saved_ += (numel * sizeof(float) - packed_size);
    
    return compressed;
}

TensorPtr ActivationCompressor::dequantize_int4(const CompressedActivation& compressed) {
    if (!compressed.metadata || compressed.metadata->mode != ActivationCompressionMode::QUANTIZE_INT4) {
        return nullptr;
    }
    
    size_t numel = compressed.metadata->original_size / sizeof(float);
    size_t packed_size = (numel + 1) / 2;
    const uint8_t* packed = compressed.compressed_data.data();
    
    // Retrieve metadata
    const float* metadata_ptr = reinterpret_cast<const float*>(packed + packed_size);
    float scale = metadata_ptr[0];
    int zero_point = static_cast<int>(metadata_ptr[1]);
    
    auto decompressed = ops::zeros(compressed.original_shape, compressed.metadata->original_dtype);
    float* output = decompressed->data<float>();
    
    // Unpack and dequantize
    for (size_t i = 0; i < numel; i += 2) {
        uint8_t byte_val = packed[i / 2];
        
        // Lower 4 bits
        int q1 = (byte_val & 0x0F) - 8;  // Map back to -8 to 7
        output[i] = (q1 + zero_point) * scale;
        
        // Upper 4 bits
        if (i + 1 < numel) {
            int q2 = ((byte_val >> 4) & 0x0F) - 8;
            output[i + 1] = (q2 + zero_point) * scale;
        }
    }
    
    return decompressed;
}

// ============================================================================
// Sparsification
// ============================================================================

std::unique_ptr<CompressedActivation> ActivationCompressor::sparsify_activation(
    const TensorPtr& activation, float threshold, size_t activation_id) {
    
    if (!activation || activation->dtype() != kFloat32) return nullptr;
    
    const float* data = activation->data<float>();
    size_t numel = static_cast<size_t>(activation->numel());
    
    // Determine sparsity threshold (absolute value)
    float abs_threshold = threshold;
    if (threshold < 1.0f) {
        // threshold is a ratio, compute actual threshold from data statistics
        std::vector<float> abs_values(numel);
        for (size_t i = 0; i < numel; ++i) {
            abs_values[i] = std::abs(data[i]);
        }
        std::sort(abs_values.begin(), abs_values.end());
        size_t threshold_idx = static_cast<size_t>(numel * threshold);
        abs_threshold = abs_values[threshold_idx];
    }
    
    // Count non-zero elements
    std::vector<size_t> indices;
    std::vector<float> values;
    indices.reserve(numel / 2);
    values.reserve(numel / 2);
    
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(data[i]) >= abs_threshold) {
            indices.push_back(i);
            values.push_back(data[i]);
        }
    }
    
    // Create compressed container
    size_t sparse_size = indices.size();
    size_t compressed_bytes = sparse_size * (sizeof(size_t) + sizeof(float)) + sizeof(size_t);
    
    auto mode = (threshold >= 0.75f) ? ActivationCompressionMode::SPARSE_75 : ActivationCompressionMode::SPARSE_50;
    auto compressed = std::make_unique<CompressedActivation>(compressed_bytes, mode);
    
    compressed->metadata->original_size = numel * sizeof(float);
    compressed->metadata->compressed_size = compressed_bytes;
    compressed->metadata->compression_ratio = 
        static_cast<float>(compressed->metadata->original_size) / compressed->metadata->compressed_size;
    compressed->metadata->sparsity_ratio = static_cast<float>(sparse_size) / numel;
    compressed->metadata->sparse_elements = sparse_size;
    compressed->original_shape = activation->shape();
    
    // Pack sparse data: [num_sparse | indices... | values...]
    compressed->compressed_data.resize(compressed_bytes);
    uint8_t* buffer = compressed->compressed_data.data();
    
    size_t* num_sparse_ptr = reinterpret_cast<size_t*>(buffer);
    *num_sparse_ptr = sparse_size;
    
    size_t* indices_ptr = reinterpret_cast<size_t*>(buffer + sizeof(size_t));
    float* values_ptr = reinterpret_cast<float*>(buffer + sizeof(size_t) + sparse_size * sizeof(size_t));
    
    std::memcpy(indices_ptr, indices.data(), sparse_size * sizeof(size_t));
    std::memcpy(values_ptr, values.data(), sparse_size * sizeof(float));
    
    total_bytes_saved_ += (numel * sizeof(float) - compressed_bytes);
    
    return compressed;
}

TensorPtr ActivationCompressor::desparsify_activation(const CompressedActivation& compressed) {
    if (!compressed.metadata) return nullptr;
    
    size_t numel = compressed.metadata->original_size / sizeof(float);
    auto decompressed = ops::zeros(compressed.original_shape, compressed.metadata->original_dtype);
    float* output = decompressed->data<float>();
    
    // Unpack sparse data
    const uint8_t* buffer = compressed.compressed_data.data();
    const size_t* num_sparse_ptr = reinterpret_cast<const size_t*>(buffer);
    size_t sparse_size = *num_sparse_ptr;
    
    const size_t* indices_ptr = reinterpret_cast<const size_t*>(buffer + sizeof(size_t));
    const float* values_ptr = reinterpret_cast<const float*>(
        buffer + sizeof(size_t) + sparse_size * sizeof(size_t));
    
    // Fill sparse values (rest are already zero from zeros())
    for (size_t i = 0; i < sparse_size; ++i) {
        size_t idx = indices_ptr[i];
        if (idx < numel) {
            output[idx] = values_ptr[i];
        }
    }
    
    return decompressed;
}

// ============================================================================
// Lossy Compression (Simple Version)
// ============================================================================

std::unique_ptr<CompressedActivation> ActivationCompressor::lossy_compress(
    const TensorPtr& activation, float target_ratio, size_t activation_id) {
    
    // Simple lossy: combine sparsification + quantization
    // First sparsify to reduce data, then quantize
    auto sparsified = sparsify_activation(activation, 1.0f - target_ratio, activation_id);
    if (!sparsified) return nullptr;
    
    sparsified->metadata->mode = ActivationCompressionMode::LOSSY_COMPRESS;
    sparsified->metadata->is_lossy = true;
    
    return sparsified;
}

TensorPtr ActivationCompressor::lossy_decompress(const CompressedActivation& compressed) {
    // Same as desparsify for simple lossy compression
    return desparsify_activation(compressed);
}

// ============================================================================
// Adaptive Compression Selection
// ============================================================================

ActivationCompressionMode ActivationCompressor::select_optimal_compression_mode(
    const TensorPtr& activation,
    const std::string& layer_name,
    bool is_critical) {
    
    if (!config_.enable_adaptive_compression) {
        // Default to INT8 if enabled
        if (config_.enable_int8_quantization) {
            return ActivationCompressionMode::QUANTIZE_INT8;
        }
        return ActivationCompressionMode::NONE;
    }
    
    // Analyze activation statistics
    const float* data = activation->data<float>();
    size_t numel = static_cast<size_t>(activation->numel());
    
    // Calculate sparsity
    size_t near_zero = 0;
    float threshold = config_.default_sparsity_threshold;
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(data[i]) < threshold) {
            near_zero++;
        }
    }
    float sparsity_ratio = static_cast<float>(near_zero) / numel;
    
    // Decision logic
    if (current_memory_pressure_ > 0.8f) {
        // High memory pressure: aggressive compression
        if (sparsity_ratio > 0.75f && config_.enable_sparsification) {
            return ActivationCompressionMode::SPARSE_75;
        }
        return ActivationCompressionMode::QUANTIZE_INT8;
    }
    
    if (sparsity_ratio > 0.6f && config_.enable_sparsification) {
        return ActivationCompressionMode::SPARSE_50;
    }
    
    if (config_.enable_int8_quantization) {
        return ActivationCompressionMode::QUANTIZE_INT8;
    }
    
    return ActivationCompressionMode::NONE;
}

float ActivationCompressor::estimate_compression_ratio(
    const TensorPtr& activation, ActivationCompressionMode mode) {
    
    switch (mode) {
        case ActivationCompressionMode::QUANTIZE_INT8:
            return 4.0f;  // FP32 -> INT8
            
        case ActivationCompressionMode::QUANTIZE_INT4:
            return 8.0f;  // FP32 -> INT4
            
        case ActivationCompressionMode::SPARSE_50:
            return 2.0f;  // 50% sparsity
            
        case ActivationCompressionMode::SPARSE_75:
            return 4.0f;  // 75% sparsity
            
        case ActivationCompressionMode::LOSSY_COMPRESS:
            return config_.lossy_compression_ratio;
            
        default:
            return 1.0f;
    }
}

// ============================================================================
// System State Management
// ============================================================================

void ActivationCompressor::update_system_state(
    float memory_pressure, int battery_level, bool is_thermal_throttling) {
    
    current_memory_pressure_ = memory_pressure;
    current_battery_level_ = battery_level;
    is_thermal_throttling_ = is_thermal_throttling;
}

void ActivationCompressor::configure_compression(const CompressionConfig& config) {
    config_ = config;
}

// ============================================================================
// Statistics and Utilities
// ============================================================================

ActivationCompressor::CompressionStats ActivationCompressor::get_compression_stats() const {
    CompressionStats stats;
    stats.total_compressions = total_compressions_;
    stats.total_decompressions = total_decompressions_;
    stats.total_bytes_saved = total_bytes_saved_;
    
    if (total_compressions_ > 0) {
        stats.average_compression_time_ms = total_compression_time_ / total_compressions_;
        stats.average_compression_ratio = static_cast<float>(total_bytes_saved_) / 
            (total_compressions_ * 1024.0f);  // Rough estimate
    } else {
        stats.average_compression_time_ms = 0.0;
        stats.average_compression_ratio = 1.0f;
    }
    
    if (total_decompressions_ > 0) {
        stats.average_decompression_time_ms = total_decompression_time_ / total_decompressions_;
    } else {
        stats.average_decompression_time_ms = 0.0;
    }
    
    stats.average_quality_score = 0.95f;  // Placeholder
    
    return stats;
}

void ActivationCompressor::clear_compressed_cache() {
    std::lock_guard<std::mutex> lock(compression_mutex_);
    compressed_cache_.clear();
}

void ActivationCompressor::update_compression_statistics(
    const CompressionMetadata& metadata, double compression_time) {
    
    (void)metadata;  // Unused for now
    double current = total_compression_time_.load();
    while (!total_compression_time_.compare_exchange_weak(current, current + compression_time));
}

float ActivationCompressor::calculate_mse_error(
    const TensorPtr& original, const TensorPtr& reconstructed) {
    
    if (!original || !reconstructed) return 0.0f;
    
    const float* orig_data = original->data<float>();
    const float* recon_data = reconstructed->data<float>();
    size_t numel = static_cast<size_t>(original->numel());
    
    float mse = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        float diff = orig_data[i] - recon_data[i];
        mse += diff * diff;
    }
    
    return mse / numel;
}

float ActivationCompressor::calculate_compression_quality(
    const TensorPtr& original, const CompressedActivation& compressed) {
    
    auto reconstructed = decompress_activation(compressed);
    if (!reconstructed) return 0.0f;
    
    float mse = calculate_mse_error(original, reconstructed);
    
    // Quality score: 1.0 = perfect, 0.0 = completely wrong
    // Use normalized MSE: quality = 1 / (1 + mse)
    return 1.0f / (1.0f + mse);
}

ActivationCompressionMode ActivationCompressor::adapt_compression_for_mobile_state(
    ActivationCompressionMode base_mode) {
    
    // Increase compression under low battery or thermal throttling
    if (is_thermal_throttling_ || current_battery_level_ < 20) {
        if (base_mode == ActivationCompressionMode::SPARSE_50) {
            return ActivationCompressionMode::SPARSE_75;
        }
        if (base_mode == ActivationCompressionMode::QUANTIZE_INT8 && config_.enable_int4_quantization) {
            return ActivationCompressionMode::QUANTIZE_INT4;
        }
    }
    
    return base_mode;
}

bool ActivationCompressor::should_use_hardware_acceleration(const TensorPtr& activation) {
    // Simple heuristic: use HW acceleration for large tensors
    return config_.enable_hardware_acceleration && 
           activation && activation->numel() > 1024;
}

void ActivationCompressor::optimize_compression_for_power_efficiency(CompressionConfig& config) {
    if (config_.optimize_for_power_efficiency) {
        config.enable_hardware_acceleration = false;  // Reduce power
        config.max_compression_threads = 1;           // Single thread
    }
}

bool ActivationCompressor::compress_with_neon_simd(
    const TensorPtr& activation, CompressedActivation& compressed) {
    
    // Placeholder for NEON SIMD acceleration
    // Will implement if needed
    return false;
}

bool ActivationCompressor::compress_with_gpu_acceleration(
    const TensorPtr& activation, CompressedActivation& compressed) {
    
    // Placeholder for GPU acceleration
    // Will implement if needed
    return false;
}

// ============================================================================
// Mobile Quantization Utilities
// ============================================================================

namespace mobile_quantization {

std::pair<float, int> calculate_quantization_params(const TensorPtr& tensor, int target_bits) {
    if (!tensor) return {1.0f, 0};
    
    const float* data = tensor->data<float>();
    size_t numel = static_cast<size_t>(tensor->numel());
    
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < numel; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    int qmax = (1 << target_bits) - 1;
    float scale = (max_val - min_val) / qmax;
    if (scale < 1e-8f) scale = 1e-8f;
    
    int zero_point = static_cast<int>(std::round(-min_val / scale));
    zero_point = std::max(0, std::min(qmax, zero_point));
    
    return {scale, zero_point};
}

TensorPtr quantize_activation_int8_mobile(const TensorPtr& activation) {
    // Simple wrapper for mobile-optimized INT8 quantization
    auto [scale, zero_point] = calculate_quantization_params(activation, 8);
    
    // Placeholder: return original for now
    // Full implementation would create INT8 tensor
    return activation;
}

TensorPtr dequantize_activation_int8_mobile(const TensorPtr& quantized_activation, 
                                            float scale, int zero_point) {
    // Placeholder
    return quantized_activation;
}

float calculate_quantization_error(const TensorPtr& original, const TensorPtr& quantized) {
    if (!original || !quantized) return 0.0f;
    
    const float* orig = original->data<float>();
    const float* quant = quantized->data<float>();
    size_t numel = static_cast<size_t>(original->numel());
    
    float mse = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        float diff = orig[i] - quant[i];
        mse += diff * diff;
    }
    
    return std::sqrt(mse / numel);  // RMSE
}

} // namespace mobile_quantization

} // namespace memory
} // namespace ops

