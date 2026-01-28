/**
 * @file mobile_optimizer_state_manager.cpp
 * @brief mobileOptimizerstatemanagerimplements
 */

#include "mobile_optimizer_state_manager.h"
#include "optimizer_utils.h"
#include "../core/logger.h"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <iostream>

// ARM NEON intrinsics for CPU optimization
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace ops {
namespace optim {

// ===============================================================================
// OptimizerStateBuffer implements
// ===============================================================================

OptimizerStateBuffer::OptimizerStateBuffer(size_t size_bytes, bool cache_align)
    : buffer_size_(size_bytes), used_size_(0), is_cache_aligned_(cache_align) {
    
    if (cache_align) {
                // [Translated]
        buffer_ptr_ = std::aligned_alloc(64, size_bytes);
    } else {
        buffer_ptr_ = std::malloc(size_bytes);
    }
    
    if (!buffer_ptr_) {
        throw std::runtime_error("Failed to allocate optimizer state buffer");
    }
    
    // initializeasallidle
    free_chunks_.emplace_back(0, size_bytes);
}

OptimizerStateBuffer::~OptimizerStateBuffer() {
    if (buffer_ptr_) {
        std::free(buffer_ptr_);
        buffer_ptr_ = nullptr;
    }
}

void* OptimizerStateBuffer::allocate(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
        // [Translated]
    size_bytes = (size_bytes + 7) & ~7;
    
        // [Translated]
    for (auto it = free_chunks_.begin(); it != free_chunks_.end(); ++it) {
        if (it->second >= size_bytes) {
            size_t offset = it->first;
            size_t remaining = it->second - size_bytes;
            
            // removeorupdateidleblock
            if (remaining > 0) {
                it->first += size_bytes;
                it->second = remaining;
            } else {
                free_chunks_.erase(it);
            }
            
            used_size_ += size_bytes;
            return static_cast<char*>(buffer_ptr_) + offset;
        }
    }
    
    return nullptr;     // [Translated]
}

void OptimizerStateBuffer::deallocate(void* ptr, size_t size_bytes) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
        // [Translated]
    size_bytes = (size_bytes + 7) & ~7;
    
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(buffer_ptr_);
    
    // addtoidlelist
    free_chunks_.emplace_back(offset, size_bytes);
    used_size_ -= size_bytes;
    
        // [Translated]
    std::sort(free_chunks_.begin(), free_chunks_.end());
    
    for (size_t i = 0; i + 1 < free_chunks_.size(); ) {
        if (free_chunks_[i].first + free_chunks_[i].second == free_chunks_[i+1].first) {
            free_chunks_[i].second += free_chunks_[i+1].second;
            free_chunks_.erase(free_chunks_.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

void OptimizerStateBuffer::defragment() {
    // [Translated comment removed - see documentation]
        // [Translated]
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // [Translated comment removed - see documentation]
        // [Translated]
}

float OptimizerStateBuffer::get_fragmentation_ratio() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (free_chunks_.empty()) return 0.0f;
    
    size_t total_free = 0;
    size_t largest_chunk = 0;
    
    for (const auto& chunk : free_chunks_) {
        total_free += chunk.second;
        largest_chunk = std::max(largest_chunk, chunk.second);
    }
    
    if (total_free == 0) return 0.0f;
    
    // [Translated comment removed - see documentation]
    return 1.0f - static_cast<float>(largest_chunk) / total_free;
}

// ===============================================================================
// OptimizerStateCompressor implements
// ===============================================================================

OptimizerStateCompressor::OptimizerStateCompressor(OptimizerStateCompression mode)
    : default_compression_(mode) {
}

std::pair<TensorPtr, float> OptimizerStateCompressor::compress(
    const TensorPtr& input,
    OptimizerStateCompression mode) {
    
    TensorPtr compressed;
    size_t original_size = input->numel() * sizeof(float);
    size_t compressed_size = 0;
    
    switch (mode) {
        case OptimizerStateCompression::FP16:
            compressed = compress_fp16(input);
            compressed_size = input->numel() * sizeof(uint16_t);
            break;
            
        case OptimizerStateCompression::INT8_QUANTIZED:
            compressed = compress_int8_quantized(input);
            compressed_size = input->numel() * sizeof(int8_t) + 2 * sizeof(float); // +scale+zero_point
            break;
            
        case OptimizerStateCompression::NONE:
        default:
            compressed = input->clone();
            compressed_size = original_size;
            break;
    }
    
    float compression_ratio = static_cast<float>(compressed_size) / original_size;
    return {compressed, compression_ratio};
}

TensorPtr OptimizerStateCompressor::decompress(
    const TensorPtr& compressed,
    OptimizerStateCompression mode) {
    
    switch (mode) {
        case OptimizerStateCompression::FP16:
            return decompress_fp16(compressed);
            
        case OptimizerStateCompression::INT8_QUANTIZED:
            return decompress_int8_quantized(compressed);
            
        case OptimizerStateCompression::NONE:
        default:
            return compressed->clone();
    }
}

std::tuple<TensorPtr, OptimizerStateCompression, float> 
OptimizerStateCompressor::adaptive_compress(const TensorPtr& input, float importance) {
    
    OptimizerStateCompression selected_mode;
    
        // [Translated]
    if (importance > 0.8f) {
        selected_mode = OptimizerStateCompression::FP16;         // [Translated]
    } else if (importance > 0.5f) {
        selected_mode = OptimizerStateCompression::INT8_QUANTIZED;         // [Translated]
    } else {
        selected_mode = OptimizerStateCompression::INT8_SPARSE;         // [Translated]
    }
    
    auto [compressed, ratio] = compress(input, selected_mode);
    return {compressed, selected_mode, ratio};
}

TensorPtr OptimizerStateCompressor::compress_fp16(const TensorPtr& input) {
    // FP32 -> FP16 compression
    size_t numel = input->numel();
    auto compressed = ops::empty({static_cast<int64_t>(numel)}, ops::kFloat16);
    
    const float* src = input->data<float>();
    uint16_t* dst = compressed->data<uint16_t>();
    
#ifdef __ARM_NEON
        // [Translated]
    compress_fp32_to_fp16_simd(src, dst, numel);
#else
    // scalarversion
    for (size_t i = 0; i < numel; ++i) {
        dst[i] = float_to_half(src[i]);
    }
#endif
    
    return compressed;
}

TensorPtr OptimizerStateCompressor::decompress_fp16(const TensorPtr& compressed) {
    // FP16 -> FP32 decompress
    size_t numel = compressed->numel();
    auto decompressed = ops::empty({static_cast<int64_t>(numel)}, ops::kFloat32);
    
    const uint16_t* src = compressed->data<uint16_t>();
    float* dst = decompressed->data<float>();
    
#ifdef __ARM_NEON
        // [Translated]
    decompress_fp16_to_fp32_simd(src, dst, numel);
#else
    // scalarversion
    for (size_t i = 0; i < numel; ++i) {
        dst[i] = half_to_float(src[i]);
    }
#endif
    
    return decompressed;
}

TensorPtr OptimizerStateCompressor::compress_int8_quantized(const TensorPtr& input) {
    // INT8quantizationcompression
    size_t numel = input->numel();
    const float* src = input->data<float>();
    
    // computescaleandzero_point
    float min_val = *std::min_element(src, src + numel);
    float max_val = *std::max_element(src, src + numel);
    
    float scale = (max_val - min_val) / 255.0f;
    float zero_point = -min_val / scale;
    
    // createquantizationtensor（storageint8 + scale + zero_point）
    auto compressed = ops::empty({static_cast<int64_t>(numel + 2)}, ops::kInt8);
    int8_t* dst = compressed->data<int8_t>();
    
    // storagescaleandzero_point
    float* meta = reinterpret_cast<float*>(dst + numel);
    meta[0] = scale;
    meta[1] = zero_point;
    
    // quantization
    for (size_t i = 0; i < numel; ++i) {
        float quantized = src[i] / scale + zero_point;
        dst[i] = static_cast<int8_t>(std::round(std::max(0.0f, std::min(255.0f, quantized)))) - 128;
    }
    
    return compressed;
}

TensorPtr OptimizerStateCompressor::decompress_int8_quantized(const TensorPtr& compressed) {
        // [Translated]
    size_t numel = compressed->numel() - 2;
    const int8_t* src = compressed->data<int8_t>();
    
    // readscaleandzero_point
    const float* meta = reinterpret_cast<const float*>(src + numel);
    float scale = meta[0];
    float zero_point = meta[1];
    
    auto decompressed = ops::empty({static_cast<int64_t>(numel)}, ops::kFloat32);
    float* dst = decompressed->data<float>();
    
        // [Translated]
    for (size_t i = 0; i < numel; ++i) {
        dst[i] = (static_cast<float>(src[i]) + 128.0f - zero_point) * scale;
    }
    
    return decompressed;
}

#ifdef __ARM_NEON
void OptimizerStateCompressor::compress_fp32_to_fp16_simd(
    const float* src, uint16_t* dst, size_t count) {
    
    // [Translated comment removed - see documentation]
    size_t simd_count = count / 4;
    
    for (size_t i = 0; i < simd_count; ++i) {
        float32x4_t v_fp32 = vld1q_f32(src + i * 4);
        float16x4_t v_fp16 = vcvt_f16_f32(v_fp32);
        vst1_u16(dst + i * 4, vreinterpret_u16_f16(v_fp16));
    }
    
    // [Translated comment removed - see documentation]
    for (size_t i = simd_count * 4; i < count; ++i) {
        dst[i] = float_to_half(src[i]);
    }
}

void OptimizerStateCompressor::decompress_fp16_to_fp32_simd(
    const uint16_t* src, float* dst, size_t count) {
    
    // [Translated comment removed - see documentation]
    size_t simd_count = count / 4;
    
    for (size_t i = 0; i < simd_count; ++i) {
        float16x4_t v_fp16 = vreinterpret_f16_u16(vld1_u16(src + i * 4));
        float32x4_t v_fp32 = vcvt_f32_f16(v_fp16);
        vst1q_f32(dst + i * 4, v_fp32);
    }
    
    // [Translated comment removed - see documentation]
    for (size_t i = simd_count * 4; i < count; ++i) {
        dst[i] = half_to_float(src[i]);
    }
}
#endif

// ===============================================================================
// OptimizerStateIOManager implements
// ===============================================================================

OptimizerStateIOManager::OptimizerStateIOManager(const std::string& path, bool compress)
    : storage_path_(path), enable_compression_(compress),
      total_io_operations_(0), total_bytes_written_(0), total_bytes_read_(0) {
    
    std::filesystem::create_directories(path);
}

std::string OptimizerStateIOManager::save_state_to_disk(
    size_t state_id,
    OptimizerStateType state_type,
    const TensorPtr& data) {
    
    std::string type_suffix;
    switch (state_type) {
        case OptimizerStateType::MOMENTUM:
            type_suffix = "momentum";
            break;
        case OptimizerStateType::VARIANCE:
            type_suffix = "variance";
            break;
        case OptimizerStateType::MASTER_WEIGHTS:
            type_suffix = "master";
            break;
        default:
            type_suffix = "unknown";
    }
    
    std::string filename = storage_path_ + "/state_" + std::to_string(state_id) + "_" + type_suffix + ".bin";
    
    auto start_time = std::chrono::steady_clock::now();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create state file: " + filename);
    }
    
    // writetensor shapeanddtype
    auto shape = data->shape();
    size_t ndim = shape.size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    file.write(reinterpret_cast<const char*>(shape.data()), sizeof(int64_t) * ndim);
    
    int dtype = static_cast<int>(data->dtype());
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    
    // writetensordata
    size_t data_size = data->numel() * DTypeUtils::size_of(data->dtype());
    file.write(reinterpret_cast<const char*>(data->data_ptr()), data_size);
    
    file.close();
    
    auto end_time = std::chrono::steady_clock::now();
        // [Translated]
    (void)end_time;      // [Translated]
    (void)start_time;
    
    total_io_operations_++;
    total_bytes_written_ += data_size;
    
    return filename;
}

TensorPtr OptimizerStateIOManager::load_state_from_disk(const std::string& path) {
    auto start_time = std::chrono::steady_clock::now();
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open state file: " + path);
    }
    
    // readtensor shapeanddtype
    size_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    std::vector<int64_t> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()), sizeof(int64_t) * ndim);
    
    int dtype_int;
    file.read(reinterpret_cast<char*>(&dtype_int), sizeof(dtype_int));
    DType dtype = static_cast<DType>(dtype_int);
    
        // [Translated]
    auto tensor = ops::empty(shape, dtype);
    size_t data_size = tensor->numel() * DTypeUtils::size_of(dtype);
    file.read(reinterpret_cast<char*>(tensor->data_ptr()), data_size);
    
    file.close();
    
    auto end_time = std::chrono::steady_clock::now();
        // [Translated]
    (void)end_time;      // [Translated]
    (void)start_time;
    
    total_io_operations_++;
    total_bytes_read_ += data_size;
    
    return tensor;
}

void OptimizerStateIOManager::delete_state_file(const std::string& path) {
    std::filesystem::remove(path);
}

OptimizerStateIOManager::IOStats OptimizerStateIOManager::get_io_stats() const {
    IOStats stats;
    stats.total_operations = total_io_operations_.load();
    stats.total_bytes_written = total_bytes_written_.load();
    stats.total_bytes_read = total_bytes_read_.load();
    
    // [Translated comment removed - see documentation]
    stats.average_write_speed_mbps = 100.0; // TODO: actualcompute
    stats.average_read_speed_mbps = 100.0;  // TODO: actualcompute
    
    return stats;
}

// ===============================================================================
// [Translated]
// ===============================================================================

MobileOptimizerStateManager::MobileOptimizerStateManager(
    const MobileOptimizerStateConfig& config,
    MobileParameterManager* param_manager)
    : config_(config),
      param_manager_(param_manager),
      active_memory_used_(0),
      standby_memory_used_(0),
      compressed_memory_used_(0),
      current_cpu_utilization_(0.0f),
      is_thermal_throttling_(false),
      is_low_battery_(false) {
    
    std::cout << "[MobileOptimizerStateManager] Initializing..." << std::endl;
    
    initialize_components();
    
    std::cout << "[MobileOptimizerStateManager] Initialized successfully" << std::endl;
    std::cout << "  Active Memory Budget: " << config_.max_active_memory_mb << "MB" << std::endl;
    std::cout << "  Standby Memory Budget: " << config_.max_standby_memory_mb << "MB" << std::endl;
    std::cout << "  Compression: " << (config_.enable_compression ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  CPU SIMD: " << (config_.enable_cpu_simd ? "Enabled" : "Disabled") << std::endl;
}

MobileOptimizerStateManager::~MobileOptimizerStateManager() {
    cleanup_components();
    std::cout << "[MobileOptimizerStateManager] Destroyed" << std::endl;
}

void MobileOptimizerStateManager::initialize_components() {
    // creatememorybuffer
    if (config_.use_contiguous_buffers) {
        active_buffer_ = std::make_unique<OptimizerStateBuffer>(
            config_.max_active_memory_mb * 1024 * 1024,
            config_.enable_cache_alignment
        );
        
        standby_buffer_ = std::make_unique<OptimizerStateBuffer>(
            config_.max_standby_memory_mb * 1024 * 1024,
            config_.enable_cache_alignment
        );
        
        std::cout << "  Created contiguous memory buffers" << std::endl;
    }
    
        // [Translated]
    if (config_.enable_compression) {
        compressor_ = std::make_unique<OptimizerStateCompressor>(config_.default_compression);
        std::cout << "  Created state compressor" << std::endl;
    }
    
    // createI/Omanager
    if (config_.enable_disk_offload) {
        io_manager_ = std::make_unique<OptimizerStateIOManager>(
            config_.storage_path,
            config_.enable_compression
        );
        std::cout << "  Created I/O manager" << std::endl;
    }
    
    // initializestatisticsinfo
    stats_ = {};
}

void MobileOptimizerStateManager::cleanup_components() {
        // [Translated]
    momentum_states_.clear();
    variance_states_.clear();
    master_weights_.clear();
    state_metadata_.clear();
    state_groups_.clear();
    
    active_buffer_.reset();
    standby_buffer_.reset();
    compressor_.reset();
    io_manager_.reset();
}

void MobileOptimizerStateManager::register_parameter_state(
    size_t param_id,
    const std::string& param_name,
    size_t param_size,
    const std::string& group_name,
    bool requires_grad) {
    
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
        // [Translated]
    auto metadata = std::make_unique<OptimizerStateMetadata>(param_id, param_name, param_size);
    metadata->requires_grad = requires_grad;
    metadata->is_trainable = requires_grad;
    
        // [Translated]
    size_t group_id;
    auto group_it = group_name_to_id_.find(group_name);
    if (group_it == group_name_to_id_.end()) {
        group_id = state_groups_.size();
        auto group = std::make_unique<OptimizerStateGroup>(group_name);
        state_groups_.push_back(std::move(group));
        group_name_to_id_[group_name] = group_id;
    } else {
        group_id = group_it->second;
    }
    
        // [Translated]
    state_groups_[group_id]->param_ids.push_back(param_id);
    param_to_group_map_[param_id] = group_id;
    
    state_metadata_[param_id] = std::move(metadata);
    
    stats_.total_states++;
    
    std::cout << "[OptimizerStateManager] Registered state for param '" << param_name 
              << "' (ID=" << param_id << ", Size=" << param_size << ", Group=" << group_name << ")" << std::endl;
}

TensorPtr MobileOptimizerStateManager::get_momentum_state(size_t param_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
        // [Translated]
    auto it = momentum_states_.find(param_id);
    if (it != momentum_states_.end()) {
        update_access_pattern(param_id);
        stats_.cache_hits++;
        
                // [Translated]
        auto metadata_it = state_metadata_.find(param_id);
        if (metadata_it != state_metadata_.end()) {
            auto& metadata = metadata_it->second;
            if (metadata->compression_mode != OptimizerStateCompression::NONE && compressor_) {
                                // [Translated]
                TensorPtr decompressed = compressor_->decompress(it->second, metadata->compression_mode);
                return decompressed;
            }
        }
        
        return it->second;
    }
    
    stats_.cache_misses++;
    
    // requireload
    load_state_internal(param_id, OptimizerStateType::MOMENTUM);
    
    return momentum_states_[param_id];
}

TensorPtr MobileOptimizerStateManager::get_variance_state(size_t param_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
        // [Translated]
    auto it = variance_states_.find(param_id);
    if (it != variance_states_.end()) {
        update_access_pattern(param_id);
        stats_.cache_hits++;
        
                // [Translated]
        auto metadata_it = state_metadata_.find(param_id);
        if (metadata_it != state_metadata_.end()) {
            auto& metadata = metadata_it->second;
            if (metadata->compression_mode != OptimizerStateCompression::NONE && compressor_) {
                                // [Translated]
                TensorPtr decompressed = compressor_->decompress(it->second, metadata->compression_mode);
                return decompressed;
            }
        }
        
        return it->second;
    }
    
    stats_.cache_misses++;
    
    // requireload
    load_state_internal(param_id, OptimizerStateType::VARIANCE);
    
    return variance_states_[param_id];
}

void MobileOptimizerStateManager::update_momentum_state(size_t param_id, const TensorPtr& new_momentum) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    momentum_states_[param_id] = new_momentum;
    
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it != state_metadata_.end()) {
        auto& metadata = metadata_it->second;
        
                // [Translated]
        size_t old_size = metadata->momentum_size_bytes;
        size_t new_size = new_momentum->numel() * sizeof(float);
        if (metadata->momentum_tier == OptimizerStateTier::COMPRESSED) {
            // [Translated comment removed - see documentation]
            active_memory_used_ += (new_size - old_size);
        }
        
        // 🔴 criticalfix: update statebackmustresetcompressioninfo
                // [Translated]
        // [Translated comment removed - see documentation]
        metadata->compression_mode = OptimizerStateCompression::NONE;
        metadata->momentum_tier = OptimizerStateTier::ACTIVE_MEMORY;
        metadata->momentum_size_bytes = new_size;
        
        metadata->is_dirty = true;
        metadata->is_loaded = true;
        update_access_pattern(param_id);
    }
}

void MobileOptimizerStateManager::update_variance_state(size_t param_id, const TensorPtr& new_variance) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    variance_states_[param_id] = new_variance;
    
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it != state_metadata_.end()) {
        auto& metadata = metadata_it->second;
        
                // [Translated]
        size_t old_size = metadata->variance_size_bytes;
        size_t new_size = new_variance->numel() * sizeof(float);
        if (metadata->variance_tier == OptimizerStateTier::COMPRESSED) {
            // [Translated comment removed - see documentation]
            active_memory_used_ += (new_size - old_size);
        }
        
        // 🔴 criticalfix: update statebackmustresetcompressioninfo
                // [Translated]
        // [Translated comment removed - see documentation]
        metadata->compression_mode = OptimizerStateCompression::NONE;
        metadata->variance_tier = OptimizerStateTier::ACTIVE_MEMORY;
        metadata->variance_size_bytes = new_size;
        
        metadata->is_dirty = true;
        metadata->is_loaded = true;
        update_access_pattern(param_id);
    }
}

void MobileOptimizerStateManager::release_parameter_state(size_t param_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it == state_metadata_.end()) {
        return;
    }
    
    // [Translated comment removed - see documentation]
    float memory_pressure = calculate_memory_pressure();
    if (memory_pressure > config_.offload_threshold) {
        offload_state_internal(param_id, OptimizerStateType::MOMENTUM);
        offload_state_internal(param_id, OptimizerStateType::VARIANCE);
    }
}

void MobileOptimizerStateManager::load_state_internal(size_t param_id, OptimizerStateType state_type) {
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it == state_metadata_.end()) {
        throw std::runtime_error("Unknown parameter ID: " + std::to_string(param_id));
    }
    
    auto& metadata = metadata_it->second;
    
        // [Translated]
    bool from_disk = false;
    std::string storage_path;
    
    if (state_type == OptimizerStateType::MOMENTUM) {
        from_disk = (metadata->momentum_tier == OptimizerStateTier::DISK_STORAGE);
        storage_path = metadata->momentum_storage_path;
    } else if (state_type == OptimizerStateType::VARIANCE) {
        from_disk = (metadata->variance_tier == OptimizerStateTier::DISK_STORAGE);
        storage_path = metadata->variance_storage_path;
    }
    
    TensorPtr state;
    
    if (from_disk && !storage_path.empty() && io_manager_) {
                // [Translated]
        state = io_manager_->load_state_from_disk(storage_path);
        stats_.total_loads++;
    } else {
        // createnewstate（initializeaszero）
        state = ops::zeros({static_cast<int64_t>(metadata->param_size)}, ops::kFloat32);
    }
    
    // storagetomemory
    if (state_type == OptimizerStateType::MOMENTUM) {
        momentum_states_[param_id] = state;
        metadata->momentum_tier = OptimizerStateTier::ACTIVE_MEMORY;
    } else if (state_type == OptimizerStateType::VARIANCE) {
        variance_states_[param_id] = state;
        metadata->variance_tier = OptimizerStateTier::ACTIVE_MEMORY;
    }
    
    metadata->is_loaded = true;
    active_memory_used_ += metadata->param_size * sizeof(float);
}

void MobileOptimizerStateManager::offload_state_internal(size_t param_id, OptimizerStateType state_type) {
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it == state_metadata_.end()) {
        return;
    }
    
    auto& metadata = metadata_it->second;
    
    TensorPtr state;
    std::string* storage_path_ptr = nullptr;
    OptimizerStateTier* tier_ptr = nullptr;
    
    if (state_type == OptimizerStateType::MOMENTUM) {
        auto it = momentum_states_.find(param_id);
        if (it == momentum_states_.end()) return;
        
        state = it->second;
        storage_path_ptr = &metadata->momentum_storage_path;
        tier_ptr = &metadata->momentum_tier;
        momentum_states_.erase(it);
    } else if (state_type == OptimizerStateType::VARIANCE) {
        auto it = variance_states_.find(param_id);
        if (it == variance_states_.end()) return;
        
        state = it->second;
        storage_path_ptr = &metadata->variance_storage_path;
        tier_ptr = &metadata->variance_tier;
        variance_states_.erase(it);
    }
    
    if (!state || !io_manager_) return;
    
        // [Translated]
    *storage_path_ptr = io_manager_->save_state_to_disk(param_id, state_type, state);
    *tier_ptr = OptimizerStateTier::DISK_STORAGE;
    
        // [Translated]
    // ifstatebeforebecompression，shouldusecompressionbacksize
    size_t actual_size = (state_type == OptimizerStateType::MOMENTUM) 
        ? metadata->momentum_size_bytes 
        : metadata->variance_size_bytes;
    
    active_memory_used_ -= actual_size;
    stats_.total_offloads++;
    stats_.offloaded_states++;
}

void MobileOptimizerStateManager::optimize_memory_usage() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    float memory_pressure = calculate_memory_pressure();
    
    if (memory_pressure > config_.offload_threshold) {
        // [Translated comment removed - see documentation]
        size_t target_reduction = (memory_pressure - config_.compression_threshold) * 
                                 config_.max_active_memory_mb * 1024 * 1024;
        
        auto states_to_offload = select_states_to_offload(target_reduction);
        
        for (size_t param_id : states_to_offload) {
            offload_state_internal(param_id, OptimizerStateType::MOMENTUM);
            offload_state_internal(param_id, OptimizerStateType::VARIANCE);
        }
        
        std::cout << "[OptimizerStateManager] Offloaded " << states_to_offload.size() 
                  << " states due to memory pressure" << std::endl;
    } else if (memory_pressure > config_.compression_threshold && config_.enable_compression) {
        // [Translated comment removed - see documentation]
        size_t target_reduction = (memory_pressure - 0.5f) * 
                                 config_.max_active_memory_mb * 1024 * 1024;
        
        auto states_to_compress = select_states_to_compress(target_reduction);
        
        for (size_t param_id : states_to_compress) {
            compress_parameter_state(param_id, config_.default_compression);
        }
        
        std::cout << "[OptimizerStateManager] Compressed " << states_to_compress.size() 
                  << " states to reduce memory usage" << std::endl;
    }
}

size_t MobileOptimizerStateManager::compress_parameter_state(
    size_t param_id,
    OptimizerStateCompression compression) {
    
    if (!compressor_) return 0;
    
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it == state_metadata_.end()) return 0;
    
    auto& metadata = metadata_it->second;
    size_t memory_saved = 0;
    
    // compressionmomentum
    // [Translated comment removed - see documentation]
    // [Translated comment removed - see documentation]
    // [Translated comment removed - see documentation]
    auto momentum_it = momentum_states_.find(param_id);
    if (momentum_it != momentum_states_.end()) {
        auto [compressed, ratio] = compressor_->compress(momentum_it->second, compression);
        
        size_t original_size = metadata->momentum_size_bytes;
        size_t compressed_size = compressed->numel() * DTypeUtils::size_of(compressed->dtype());
        
        momentum_states_[param_id] = compressed;
        metadata->momentum_size_bytes = compressed_size;
        metadata->compression_ratio = ratio;
        metadata->compression_mode = compression;
        metadata->momentum_tier = OptimizerStateTier::COMPRESSED;          // [Translated]
        
        memory_saved += (original_size - compressed_size);
                // [Translated]
        active_memory_used_ -= (original_size - compressed_size);
    }
    
    // compressionvariance
    // [Translated comment removed - see documentation]
    // [Translated comment removed - see documentation]
    auto variance_it = variance_states_.find(param_id);
    if (variance_it != variance_states_.end()) {
        auto [compressed, ratio] = compressor_->compress(variance_it->second, compression);
        
        size_t original_size = metadata->variance_size_bytes;
        size_t compressed_size = compressed->numel() * DTypeUtils::size_of(compressed->dtype());
        
        variance_states_[param_id] = compressed;
        metadata->variance_size_bytes = compressed_size;
        metadata->variance_tier = OptimizerStateTier::COMPRESSED;          // [Translated]
        
        memory_saved += (original_size - compressed_size);
                // [Translated]
        active_memory_used_ -= (original_size - compressed_size);
    }
    
    stats_.total_compressions++;
    stats_.compressed_states++;
    stats_.memory_saved_by_compression += memory_saved;
    
    return memory_saved;
}

size_t MobileOptimizerStateManager::offload_parameter_state(size_t param_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it == state_metadata_.end()) return 0;
    
    size_t memory_freed = 0;
    
    offload_state_internal(param_id, OptimizerStateType::MOMENTUM);
    offload_state_internal(param_id, OptimizerStateType::VARIANCE);
    
    memory_freed = metadata_it->second->param_size * sizeof(float) * 2; // momentum + variance
    
    return memory_freed;
}

size_t MobileOptimizerStateManager::calculate_memory_pressure() const {
    size_t active_used = active_memory_used_.load();
    size_t active_limit = config_.max_active_memory_mb * 1024 * 1024;
    
    return active_used / static_cast<float>(active_limit);
}

std::vector<size_t> MobileOptimizerStateManager::select_states_to_compress(size_t target_memory_reduction) {
    std::vector<std::pair<size_t, float>> candidates; // param_id, priority
    
    for (const auto& [param_id, metadata] : state_metadata_) {
        if (metadata->compression_mode == OptimizerStateCompression::NONE) {
                        // [Translated]
            float priority = static_cast<float>(metadata->access_count) / 
                           (metadata->param_size + 1.0f);
            candidates.emplace_back(param_id, priority);
        }
    }
    
    // byprioritysort
    std::sort(candidates.begin(), candidates.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; });
    
        // [Translated]
    std::vector<size_t> selected;
    size_t accumulated_reduction = 0;
    
    for (const auto& [param_id, _] : candidates) {
        selected.push_back(param_id);
        
        auto metadata_it = state_metadata_.find(param_id);
        if (metadata_it != state_metadata_.end()) {
                        // [Translated]
            accumulated_reduction += metadata_it->second->param_size * sizeof(float);
        }
        
        if (accumulated_reduction >= target_memory_reduction) {
            break;
        }
    }
    
    return selected;
}

std::vector<size_t> MobileOptimizerStateManager::select_states_to_offload(size_t target_memory_reduction) {
    std::vector<std::pair<size_t, float>> candidates; // param_id, priority
    
    for (const auto& [param_id, metadata] : state_metadata_) {
        if (metadata->is_loaded && !metadata->is_dirty) {
            // [Translated comment removed - see documentation]
            float priority = static_cast<float>(metadata->access_count) * metadata->priority;
            candidates.emplace_back(param_id, priority);
        }
    }
    
    // byprioritysort
    std::sort(candidates.begin(), candidates.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; });
    
        // [Translated]
    std::vector<size_t> selected;
    size_t accumulated_reduction = 0;
    
    for (const auto& [param_id, _] : candidates) {
        selected.push_back(param_id);
        
        auto metadata_it = state_metadata_.find(param_id);
        if (metadata_it != state_metadata_.end()) {
            accumulated_reduction += metadata_it->second->param_size * sizeof(float) * 2; // momentum + variance
        }
        
        if (accumulated_reduction >= target_memory_reduction) {
            break;
        }
    }
    
    return selected;
}

void MobileOptimizerStateManager::update_access_pattern(size_t param_id) {
    auto metadata_it = state_metadata_.find(param_id);
    if (metadata_it != state_metadata_.end()) {
        metadata_it->second->access_count++;
        metadata_it->second->last_access_time = std::chrono::steady_clock::now();
    }
}

void MobileOptimizerStateManager::update_mobile_state(
    float cpu_util,
    bool is_thermal_throttle,
    bool is_low_battery) {
    
    current_cpu_utilization_ = cpu_util;
    is_thermal_throttling_ = is_thermal_throttle;
    is_low_battery_ = is_low_battery;
    
    // according tomobilestateadjuststrategy
    if (is_thermal_throttle) {
        apply_thermal_optimization();
    }
    
    if (is_low_battery) {
        apply_battery_optimization();
    }
    
    if (cpu_util < config_.cpu_utilization_target) {
        apply_cpu_optimization();
    }
}

void MobileOptimizerStateManager::apply_cpu_optimization() {
        // [Translated]
    // [Translated comment removed - see documentation]
}

void MobileOptimizerStateManager::apply_thermal_optimization() {
    // [Translated comment removed - see documentation]
        // [Translated]
}

void MobileOptimizerStateManager::apply_battery_optimization() {
    // [Translated comment removed - see documentation]
    // [Translated comment removed - see documentation]
}

OptimizerStateStats MobileOptimizerStateManager::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
        // [Translated]
    OptimizerStateStats current_stats = stats_;
    current_stats.active_memory_used = active_memory_used_.load();
    current_stats.standby_memory_used = standby_memory_used_.load();
    current_stats.compressed_memory_used = compressed_memory_used_.load();
    
    if (current_stats.cache_hits + current_stats.cache_misses > 0) {
        current_stats.cache_hit_ratio = static_cast<float>(current_stats.cache_hits) / 
                                       (current_stats.cache_hits + current_stats.cache_misses);
    }
    
    return current_stats;
}

const OptimizerStateMetadata* MobileOptimizerStateManager::get_state_metadata(size_t param_id) const {
    auto it = state_metadata_.find(param_id);
    if (it != state_metadata_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void MobileOptimizerStateManager::save_checkpoint(const std::string& checkpoint_path) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    std::filesystem::create_directories(std::filesystem::path(checkpoint_path).parent_path());
    
    // savealloptimizerstate
        // [Translated]
    std::cout << "[MobileOptimizerStateManager] Checkpoint save at: " << checkpoint_path << std::endl;
}

void MobileOptimizerStateManager::load_checkpoint(const std::string& checkpoint_path) {
    (void)checkpoint_path;  // TODO: implements
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    // loadoptimizerstate
        // [Translated]
}

// ===============================================================================
// [Translated]
// ===============================================================================

std::unique_ptr<MobileOptimizerStateManager> create_mobile_optimizer_state_manager(
    size_t available_memory_mb,
    const std::string& storage_path,
    MobileParameterManager* param_manager) {
    
    MobileOptimizerStateConfig config;
    config.max_active_memory_mb = available_memory_mb / 2;
    config.max_standby_memory_mb = available_memory_mb;
    config.storage_path = storage_path;
    
    // according toavailablememoryadjustconfiguration
    if (available_memory_mb < 256) {
                // [Translated]
        config.default_compression = OptimizerStateCompression::INT8_QUANTIZED;
        config.compression_threshold = 0.6f;
        config.offload_threshold = 0.7f;
    } else if (available_memory_mb < 512) {
                // [Translated]
        config.default_compression = OptimizerStateCompression::FP16;
        config.compression_threshold = 0.7f;
        config.offload_threshold = 0.8f;
    } else {
                // [Translated]
        config.default_compression = OptimizerStateCompression::FP16;
        config.enable_compression = true;
        config.compression_threshold = 0.8f;
    }
    
    return std::make_unique<MobileOptimizerStateManager>(config, param_manager);
}

} // namespace memory
} // namespace ops

