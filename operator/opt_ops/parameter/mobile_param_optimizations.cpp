/**
 * @file mobile_param_optimizations.cpp
 * @brief Implementation of mobile parameter optimization classes
 * 
 * This file providess stub implementsations of the critical optimization classes
 * to fix linking issues. These are minimal working implementsations that can
 * be extended later with full functionality.
 */

#include "mobile_param_optimizations.h"
#include <cstring>
#include <cmath>
#include "../core/dtype.h"
#include <iostream>
#include <algorithm>

namespace ops {
namespace memory {

// ===============================
// ParameterQuantizer Implementation
// ===============================

ParameterQuantizer::ParameterQuantizer(QuantizationMode mode) 
    : mode_(mode), quantization_error_threshold_(0.01f) {
    std::cout << "[ParameterQuantizer] Initialized with mode: " << static_cast<int>(mode) << std::endl;
}

std::pair<void*, float> ParameterQuantizer::quantize_parameter(size_t param_id, const void* data, size_t size) {
    if (!data || size == 0) return {nullptr, 1.0f};
    
    // Implement different quantization based on mode
    void* quantized_data = nullptr;
    float compression_ratio = 1.0f;
    size_t quantized_size = size;
    
    switch (mode_) {
        case QuantizationMode::NONE:
            // No quantization - just copy the data
            quantized_data = std::malloc(size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, size);
                compression_ratio = 1.0f;
                quantized_size = size;
            }
            break;
            
        case QuantizationMode::FP16:
            // Simulate FP16 quantization (half precision)
            quantized_size = size / 2;
            quantized_data = std::malloc(quantized_size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, quantized_size);
                compression_ratio = 0.5f;
            }
            break;
            
        case QuantizationMode::INT8:
            // Simulate INT8 quantization (8-bit integer)
            quantized_size = size / 4; // Assume FP32 to INT8
            quantized_data = std::malloc(quantized_size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, quantized_size);
                compression_ratio = 0.25f;
            }
            break;
            
        case QuantizationMode::INT4:
            // Simulate INT4 quantization (4-bit integer)
            quantized_size = size / 8; // Assume FP32 to INT4
            quantized_data = std::malloc(quantized_size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, quantized_size);
                compression_ratio = 0.125f;
            }
            break;
            
        case QuantizationMode::DYNAMIC:
            // Dynamic quantization - analyze data first (simplified)
            quantized_size = size / 3; // Moderate compression
            quantized_data = std::malloc(quantized_size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, quantized_size);
                compression_ratio = 0.33f;
            }
            break;
            
        case QuantizationMode::MIXED:
            // Mixed precision - balanced approach
            quantized_size = size * 3 / 4; // 75% of original size
            quantized_data = std::malloc(quantized_size);
            if (quantized_data) {
                std::memcpy(quantized_data, data, quantized_size);
                compression_ratio = 0.75f;
            }
            break;
            
        default:
            return {nullptr, 1.0f};
    }
    
    if (!quantized_data) return {nullptr, 1.0f};
    
    // Store quantization parameters
    param_scales_[param_id] = 1.0f;
    param_zero_points_[param_id] = 0;
    param_compression_ratios_[param_id] = compression_ratio;
    
    std::cout << "[ParameterQuantizer] Quantized parameter " << param_id 
              << " (mode: " << static_cast<int>(mode_) << ") "
              << "with " << compression_ratio << " compression ratio" << std::endl;
    
    return {quantized_data, compression_ratio};
}

void ParameterQuantizer::dequantize_parameter(size_t param_id, const void* quantized_data, void* output_data, size_t size) {
    if (!quantized_data || !output_data || size == 0) return;
    
    // Simple stub implementsation - simulate dequantization
    std::memset(output_data, 0, size);
    
    // Use stored quantization parameters
    float scale = 1.0f;
    int32_t zero_point = 0;
    
    auto scale_it = param_scales_.find(param_id);
    auto zp_it = param_zero_points_.find(param_id);
    
    if (scale_it != param_scales_.end()) scale = scale_it->second;
    if (zp_it != param_zero_points_.end()) zero_point = zp_it->second;
    
    std::cout << "[ParameterQuantizer] Dequantized parameter " << param_id 
              << " with scale=" << scale << ", zero_point=" << zero_point << std::endl;
}

float ParameterQuantizer::get_compression_ratio(size_t param_id) const {
    // Return actual stored compression ratio
    auto it = param_compression_ratios_.find(param_id);
    if (it != param_compression_ratios_.end()) {
        return it->second;
    }
    
    // Default compression ratio if parameter not found
    return 1.0f; // No compression by default
}

QuantizationMode ParameterQuantizer::determine_optimal_quantization(const TensorPtr& tensor) {
    if (!tensor) return QuantizationMode::NONE;
    
    size_t tensor_size = tensor->numel();
    
    // Simple heuristic: smaller tensors can use more aggressive quantization
    if (tensor_size < 1000) return QuantizationMode::INT4;
    else if (tensor_size < 10000) return QuantizationMode::INT8;
    else return QuantizationMode::FP16;
}

// ===============================
// PinnedMemoryManager Implementation
// ===============================

PinnedMemoryManager::PinnedMemoryManager(size_t max_pinned_mb) 
    : total_pinned_memory_(0), max_pinned_memory_(max_pinned_mb * 1024 * 1024) {
    std::cout << "[PinnedMemoryManager] Initialized with max " << max_pinned_mb << "MB pinned memory" << std::endl;
}

PinnedMemoryManager::~PinnedMemoryManager() {
    // Free all pinned buffers
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    for (void* buffer : pinned_buffers_) {
        if (buffer) {
            std::free(buffer);
        }
    }
    std::cout << "[PinnedMemoryManager] Destroyed" << std::endl;
}

void* PinnedMemoryManager::allocate_pinned(size_t size) {
    // Reject zero-size allocations
    if (size == 0) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    if (total_pinned_memory_ + size > max_pinned_memory_) {
        return nullptr; // Not enough space
    }
    
    void* buffer = std::aligned_alloc(64, size);
    if (!buffer) return nullptr;
    
    pinned_buffers_.push_back(buffer);
    buffer_sizes_.push_back(size);
    buffer_in_use_.push_back(true);
    total_pinned_memory_ += size;
    
    return buffer;
}

void PinnedMemoryManager::free_pinned(void* ptr) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    for (size_t i = 0; i < pinned_buffers_.size(); ++i) {
        if (pinned_buffers_[i] == ptr) {
            std::free(ptr);
            total_pinned_memory_ -= buffer_sizes_[i];
            
            pinned_buffers_[i] = nullptr;
            buffer_sizes_[i] = 0;
            buffer_in_use_[i] = false;
            return;
        }
    }
}

size_t PinnedMemoryManager::get_available_pinned_memory() const {
    return max_pinned_memory_ - total_pinned_memory_;
}

// ===============================
// ContiguousMemoryAllocator Implementation
// ===============================

ContiguousMemoryAllocator::ContiguousMemoryAllocator(size_t total_size, size_t alignment) 
    : memory_buffer_(nullptr), total_size_(total_size), total_free_(total_size), 
      largest_contiguous_(total_size), max_allocated_(0), default_alignment_(alignment) {
    
    memory_buffer_ = std::aligned_alloc(alignment, total_size);
    if (!memory_buffer_) {
        throw std::runtime_error("Failed to allocate contiguous memory pool");
    }
    
    // Initialize with one large free block
    MemoryBlock initial_block(0, total_size, true, alignment);
    
    memory_blocks_.push_back(initial_block);
    free_blocks_by_size_[total_size].push_back(0);
    
    std::cout << "[ContiguousMemoryAllocator] Allocated " << total_size / (1024*1024) 
              << "MB contiguous memory pool with " << alignment << "-byte alignment" << std::endl;
}

ContiguousMemoryAllocator::~ContiguousMemoryAllocator() {
    if (memory_buffer_) {
        std::free(memory_buffer_);
    }
    std::cout << "[ContiguousMemoryAllocator] Destroyed. Max allocated: " 
              << max_allocated_ / (1024*1024) << "MB" << std::endl;
}

void* ContiguousMemoryAllocator::allocate(size_t size, size_t alignment) {
    // Reject zero-size allocations
    if (size == 0) {
        return nullptr;
    }
    
    if (alignment == 0) alignment = default_alignment_;
    
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    
    // Find suitable free block
    for (size_t i = 0; i < memory_blocks_.size(); ++i) {
        MemoryBlock& block = memory_blocks_[i];
        
        if (block.is_free && block.size >= size) {
            // Mark as allocated
            block.is_free = false;
            
            // Update tracking
            void* ptr = static_cast<char*>(memory_buffer_) + block.start_offset;
            pointer_to_block_[ptr] = i;
            
            total_free_ -= block.size;
            size_t allocated_size = total_size_ - total_free_;
            if (allocated_size > max_allocated_) {
                max_allocated_ = allocated_size;
            }
            
            // If block is larger than needed, split it
            if (block.size > size + alignment) {
                MemoryBlock new_free_block(block.start_offset + size, block.size - size, true, alignment);
                
                memory_blocks_.push_back(new_free_block);
                free_blocks_by_size_[new_free_block.size].push_back(memory_blocks_.size() - 1);
                
                block.size = size;
            }
            
            return ptr;
        }
    }
    
    return nullptr; // No suitable block found
}

void ContiguousMemoryAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    
    auto it = pointer_to_block_.find(ptr);
    if (it != pointer_to_block_.end()) {
        size_t block_index = it->second;
        memory_blocks_[block_index].is_free = true;
        total_free_ += memory_blocks_[block_index].size;
        
        pointer_to_block_.erase(it);
        
        // Add to free blocks
        size_t block_size = memory_blocks_[block_index].size;
        free_blocks_by_size_[block_size].push_back(block_index);
    }
}

size_t ContiguousMemoryAllocator::defragment(size_t min_block_size) {
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    
    // Simple defragmentation: compact free blocks
    std::vector<MemoryBlock> new_blocks;
    
    for (const auto& block : memory_blocks_) {
        if (!block.is_free) {
            new_blocks.push_back(block);
        }
    }
    
    // Rebuild free space as one large block if possible
    if (total_free_ > 0) {
        MemoryBlock free_block(total_size_ - total_free_, total_free_, true, default_alignment_);
        new_blocks.push_back(free_block);
    }
    
    memory_blocks_ = std::move(new_blocks);
    largest_contiguous_ = total_free_;
    
    std::cout << "[ContiguousMemoryAllocator] Defragmentation completed. Free: " 
              << total_free_ / (1024*1024) << "MB" << std::endl;
    
    (void)min_block_size; // Suppress warning for now
    return total_free_; // Return amount of free memory
}

// ===============================
// ParameterPersistenceManager Implementation
// ===============================

ParameterPersistenceManager::ParameterPersistenceManager(size_t param_threshold, size_t model_threshold) 
    : param_persistence_threshold_(param_threshold), model_persistence_threshold_(model_threshold),
      total_persistent_memory_size_(0) {
    std::cout << "[ParameterPersistenceManager] Initialized with param_threshold=" 
              << param_threshold << ", model_threshold=" << model_threshold << std::endl;
}

bool ParameterPersistenceManager::should_be_persistent(size_t param_size) const {
    return param_size <= param_persistence_threshold_ && 
           (total_persistent_memory_size_ + param_size) <= model_persistence_threshold_;
}

bool ParameterPersistenceManager::register_persistent_param(size_t param_id, size_t param_size) {
    if (should_be_persistent(param_size)) {
        persistent_params_.insert(param_id);
        total_persistent_memory_size_ += param_size;
        
        std::cout << "[ParameterPersistenceManager] Registered persistent parameter " 
                  << param_id << " (" << param_size << " bytes)" << std::endl;
        return true;
    }
    return false;
}

bool ParameterPersistenceManager::is_persistent(size_t param_id) const {
    return persistent_params_.find(param_id) != persistent_params_.end();
}

// ===============================
// ParameterReuseTracker Implementation
// ===============================

ParameterReuseTracker::ParameterReuseTracker(size_t max_reuse_distance) 
    : max_reuse_distance_(max_reuse_distance), current_step_id_(0) {
    std::cout << "[ParameterReuseTracker] Initialized with max_reuse_distance=" << max_reuse_distance << std::endl;
}

void ParameterReuseTracker::record_access(size_t param_id) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    current_step_id_++;
    auto now = std::chrono::steady_clock::now();
    
    // Update access record
    AccessRecord& record = access_records_[param_id];
    record.param_id = param_id;
    record.step_id = current_step_id_;
    record.access_count++;
    record.last_access_time = now;
    
    recent_accesses_.push_back(param_id);
    
    // Keep only recent accesses (limit to 1000 for memory efficiency)
    if (recent_accesses_.size() > 1000) {
        recent_accesses_.pop_front();
    }
    
    // Cleanup old records periodically
    if (current_step_id_ % 10000 == 0) {
        cleanup_old_records();
    }
}

bool ParameterReuseTracker::should_release_parameter(size_t param_id) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    auto it = access_records_.find(param_id);
    if (it == access_records_.end()) return true; // Not accessed yet, can release
    
    const AccessRecord& record = it->second;
    
    // Simple heuristic: release if not accessed recently
    size_t steps_since_access = current_step_id_ - record.step_id;
    return steps_since_access > max_reuse_distance_;
}

std::vector<size_t> ParameterReuseTracker::get_releasable_parameters(size_t max_count) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    std::vector<std::pair<size_t, size_t>> candidates; // (param_id, steps_since_access)
    
    for (const auto& pair : access_records_) {
        size_t param_id = pair.first;
        const AccessRecord& record = pair.second;
        size_t steps_since_access = current_step_id_ - record.step_id;
        
        if (steps_since_access > max_reuse_distance_) {
            candidates.emplace_back(param_id, steps_since_access);
        }
    }
    
    // Sort by steps since access (oldest first)
    std::sort(candidates.begin(), candidates.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(max_count, candidates.size()); ++i) {
        result.push_back(candidates[i].first);
    }
    
    return result;
}

std::vector<size_t> ParameterReuseTracker::predict_next_accesses(size_t current_param_id, size_t max_predictions) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    // Simple prediction based on recent access patterns
    std::vector<size_t> predictions;
    
    for (auto it = recent_accesses_.rbegin(); 
         it != recent_accesses_.rend() && predictions.size() < max_predictions; ++it) {
        if (*it != current_param_id && 
            std::find(predictions.begin(), predictions.end(), *it) == predictions.end()) {
            predictions.push_back(*it);
        }
    }
    
    return predictions;
}

ParameterReuseTracker::ReuseStats ParameterReuseTracker::get_reuse_stats(size_t param_id) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    ReuseStats stats;
    auto it = access_records_.find(param_id);
    
    if (it != access_records_.end()) {
        const AccessRecord& record = it->second;
        stats.total_accesses = record.access_count;
        stats.average_reuse_distance = calculate_reuse_distance(param_id);
        
        // Calculate average reuse time (simplified)
        auto now = std::chrono::steady_clock::now();
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - record.last_access_time);
        stats.average_reuse_time = time_since_last;
        
        stats.is_frequently_accessed = record.access_count > 10; // Simple threshold
    } else {
        stats.total_accesses = 0;
        stats.average_reuse_distance = 0.0;
        stats.average_reuse_time = std::chrono::milliseconds(0);
        stats.is_frequently_accessed = false;
    }
    
    return stats;
}

size_t ParameterReuseTracker::calculate_reuse_distance(size_t param_id) const {
    auto it = access_records_.find(param_id);
    if (it == access_records_.end()) return 0;
    
    return current_step_id_ - it->second.step_id;
}

void ParameterReuseTracker::update_access_patterns() {
    // This could implements more sophisticated pattern analysis
    // For now, it's a placeholder
}

void ParameterReuseTracker::cleanup_old_records() {
    // Remove records that are too old
    size_t cleanup_threshold = max_reuse_distance_ * 2;
    
    for (auto it = access_records_.begin(); it != access_records_.end();) {
        if (current_step_id_ - it->second.step_id > cleanup_threshold) {
            it = access_records_.erase(it);
        } else {
            ++it;
        }
    }
}

// ===============================
// MobilePlatforOptimizer Implementation
// ===============================

MobilePlatforOptimizer::MobilePlatforOptimizer() 
    : is_low_power_mode_(false),
      temperature_threshold_(80.0f),
      current_temperature_(25.0f),
      battery_level_(75), // 75% battery level
      thermal_monitoring_active_(false) {  // Initialize to false first
    
    last_thermal_check_ = std::chrono::steady_clock::now();
    
    std::cout << "[MobilePlatforOptimizer] Initialized" << std::endl;
    
    // Start thermal monitoring thread AFTER object is fully constructed
    thermal_monitoring_active_ = true;
    if (thermal_monitoring_active_) {
        thermal_monitor_thread_ = std::thread(&MobilePlatforOptimizer::monitor_thermal_state, this);
    }
}

MobilePlatforOptimizer::~MobilePlatforOptimizer() {
    thermal_monitoring_active_ = false;
    if (thermal_monitor_thread_.joinable()) {
        thermal_monitor_thread_.join();
    }
    std::cout << "[MobilePlatforOptimizer] Destroyed" << std::endl;
}

bool MobilePlatforOptimizer::should_throttle_operations() const {
    // Check thermal constraints
    if (current_temperature_ > temperature_threshold_) {
        return true;
    }
    
    // Check battery constraints
    if (battery_level_ < 20) {
        return true;
    }
    
    // Check low power mode
    if (is_low_power_mode_) {
        return true;
    }
    
    return false;
}

size_t MobilePlatforOptimizer::get_recommended_memory_limit() const {
    size_t base_limit = 1024 * 1024 * 1024; // 1GB base limit
    
    // Reduce based on thermal state
    if (current_temperature_ > temperature_threshold_) {
        base_limit = static_cast<size_t>(base_limit * 0.7); // 70% when overheating
    }
    
    // Reduce based on battery level
    if (battery_level_ < 30) {
        base_limit = static_cast<size_t>(base_limit * 0.8); // 80% when low battery
    }
    
    // Reduce in low power mode
    if (is_low_power_mode_) {
        base_limit = static_cast<size_t>(base_limit * 0.6); // 60% in low power mode
    }
    
    return base_limit;
}

size_t MobilePlatforOptimizer::get_recommended_batch_size(size_t base_batch_size) const {
    size_t recommended = base_batch_size;
    
    // Reduce batch size under thermal stress
    if (current_temperature_ > temperature_threshold_) {
        recommended = static_cast<size_t>(recommended * 0.5);
    }
    
    // Reduce batch size on low battery
    if (battery_level_ < 20) {
        recommended = static_cast<size_t>(recommended * 0.7);
    }
    
    return std::max(size_t(1), recommended); // At least batch size of 1
}

void MobilePlatforOptimizer::enable_low_power_mode(bool enable) {
    is_low_power_mode_ = enable;
    std::cout << "[MobilePlatforOptimizer] Low power mode: " 
              << (enable ? "ENABLED" : "DISABLED") << std::endl;
}

void MobilePlatforOptimizer::neon_memcpy(void* dest, const void* src, size_t size) {
    // Stub implementsation - use standard memcpy
    // In real implementsation, this would use ARM NEON intrinsics
    std::memcpy(dest, src, size);
}

void MobilePlatforOptimizer::neon_quantize_fp32_to_int8(const float* input, int8_t* output, 
                                                         size_t count, float scale, int32_t zero_point) {
    // Stub implementsation - simple quantization
    // In real implementsation, this would use ARM NEON intrinsics for vectorized quantization
    for (size_t i = 0; i < count; ++i) {
        float scaled = input[i] / scale;
        int32_t quantized = static_cast<int32_t>(std::round(scaled)) + zero_point;
        quantized = std::max(-128, std::min(127, quantized));
        output[i] = static_cast<int8_t>(quantized);
    }
}

void MobilePlatforOptimizer::monitor_thermal_state() {
    while (thermal_monitoring_active_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Check every second
        
        current_temperature_ = get_cpu_temperature();
        battery_level_ = get_battery_level();
        last_thermal_check_ = std::chrono::steady_clock::now();
        
        adjust_perforance_based_on_thermal_state();
    }
}

float MobilePlatforOptimizer::get_cpu_temperature() const {
    // Stub implementsation - simulate temperature variations
    static float base_temp = 25.0f;
    static float temp_trend = 0.1f;
    
    base_temp += temp_trend;
    
    if (base_temp > 85.0f) {
        temp_trend = -0.2f;
    } else if (base_temp < 20.0f) {
        temp_trend = 0.1f;
    }
    
    return std::max(20.0f, std::min(90.0f, base_temp));
}

size_t MobilePlatforOptimizer::get_battery_level() const {
    // Stub implementsation - simulate battery drain
    static size_t battery = 75;
    static int drain_direction = -1;
    
    battery += drain_direction;
    if (battery <= 10) drain_direction = 1;
    if (battery >= 95) drain_direction = -1;
    
    return std::max(size_t(5), std::min(size_t(100), battery));
}

void MobilePlatforOptimizer::adjust_perforance_based_on_thermal_state() {
    // Adjust thresholds based on current state
    if (current_temperature_ > temperature_threshold_) {
        // Thermal throttling - could adjust clock speeds, etc.
        std::cout << "[MobilePlatforOptimizer] Thermal throttling active at " 
                  << current_temperature_ << "°C" << std::endl;
    }
}

// ===============================
// AdvancedPrefetchSystem Implementation
// ===============================

AdvancedPrefetchSystem::AdvancedPrefetchSystem(size_t bucket_size) 
    : prefetch_bucket_size_(bucket_size), 
      max_prefetch_buckets_(10), // Default to 10 buckets
      prefetch_active_(false) {  // Keep false during construction
    
    // Initialize prefetch buckets
    prefetch_buckets_.reserve(max_prefetch_buckets_);
    
    std::cout << "[AdvancedPrefetchSystem] Initialized with bucket_size=" 
              << bucket_size / (1024*1024) << "MB, max_buckets=" << max_prefetch_buckets_ << std::endl;
    
    // Start prefetch worker thread AFTER object is fully constructed
    prefetch_active_ = true;
    prefetch_worker_thread_ = std::thread(&AdvancedPrefetchSystem::prefetch_worker_loop, this);
}

AdvancedPrefetchSystem::~AdvancedPrefetchSystem() {
    prefetch_active_ = false;
    prefetch_cv_.notify_all();
    
    if (prefetch_worker_thread_.joinable()) {
        prefetch_worker_thread_.join();
    }
    
    std::cout << "[AdvancedPrefetchSystem] Destroyed with " << prefetch_buckets_.size() 
              << " active buckets" << std::endl;
}

void AdvancedPrefetchSystem::schedule_prefetch(const std::vector<size_t>& param_ids, int priority) {
    if (!prefetch_active_) return;
    
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    
    // Add parameters to prefetch queue
    for (size_t param_id : param_ids) {
        prefetch_queue_.push(param_id);
    }
    
    // Organize into buckets
    organize_prefetch_buckets();
    
    // Notify worker thread
    prefetch_cv_.notify_one();
    
    (void)priority; // Suppress warning - could be used for priority queue in future
}

void AdvancedPrefetchSystem::cancel_prefetch(const std::vector<size_t>& param_ids) {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    
    // Remove from buckets that are not yet executing
    for (auto& bucket : prefetch_buckets_) {
        if (!bucket.is_prefetching) {
            for (size_t param_id : param_ids) {
                auto it = std::find(bucket.param_ids.begin(), bucket.param_ids.end(), param_id);
                if (it != bucket.param_ids.end()) {
                    bucket.param_ids.erase(it);
                }
            }
        }
    }
    
    // Remove empty buckets
    prefetch_buckets_.erase(
        std::remove_if(prefetch_buckets_.begin(), prefetch_buckets_.end(),
                      [](const PrefetchBucket& bucket) { return bucket.param_ids.empty(); }),
        prefetch_buckets_.end());
}

bool AdvancedPrefetchSystem::is_prefetching(size_t param_id) const {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    
    for (const auto& bucket : prefetch_buckets_) {
        if (bucket.is_prefetching) {
            auto it = std::find(bucket.param_ids.begin(), bucket.param_ids.end(), param_id);
            if (it != bucket.param_ids.end()) {
                return true;
            }
        }
    }
    
    return false;
}

void AdvancedPrefetchSystem::wait_for_prefetch(const std::vector<size_t>& param_ids, 
                                               std::chrono::milliseconds timeout) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - start_time < timeout) {
        bool all_completed = true;
        
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            for (size_t param_id : param_ids) {
                if (is_prefetching(param_id)) {
                    all_completed = false;
                    break;
                }
            }
        }
        
        if (all_completed) break;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

AdvancedPrefetchSystem::PrefetchStats AdvancedPrefetchSystem::get_prefetch_stats() const {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    
    PrefetchStats stats;
    stats.total_prefetches = 0;
    stats.successful_prefetches = 0;
    stats.failed_prefetches = 0;
    stats.average_prefetch_time_ms = 0;
    stats.prefetch_hit_rate = 0.0;
    
    // Count completed buckets for stats (simplified)
    for (const auto& bucket : prefetch_buckets_) {
        if (!bucket.is_prefetching) {
            stats.total_prefetches += bucket.param_ids.size();
            stats.successful_prefetches += bucket.param_ids.size(); // Assume success for now
        }
    }
    
    if (stats.total_prefetches > 0) {
        stats.prefetch_hit_rate = static_cast<double>(stats.successful_prefetches) / stats.total_prefetches;
    }
    
    return stats;
}

void AdvancedPrefetchSystem::prefetch_worker_loop() {
    while (prefetch_active_) {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        
        prefetch_cv_.wait(lock, [this] { 
            return !prefetch_active_ || !prefetch_queue_.empty() || 
                   std::any_of(prefetch_buckets_.begin(), prefetch_buckets_.end(),
                               [](const PrefetchBucket& b) { return !b.is_prefetching && !b.param_ids.empty(); });
        });
        
        if (!prefetch_active_) break;
        
        // Find bucket ready for execution
        for (auto& bucket : prefetch_buckets_) {
            if (!bucket.is_prefetching && !bucket.param_ids.empty()) {
                bucket.is_prefetching = true;
                bucket.prefetch_start_time = std::chrono::steady_clock::now();
                
                lock.unlock();
                execute_bucket_prefetch(bucket);
                lock.lock();
                
                bucket.is_prefetching = false;
                bucket.param_ids.clear(); // Mark as completed
                break;
            }
        }
    }
}

void AdvancedPrefetchSystem::organize_prefetch_buckets() {
    // Simple organization: create buckets based on queue contents
    while (!prefetch_queue_.empty() && prefetch_buckets_.size() < max_prefetch_buckets_) {
        PrefetchBucket bucket;
        bucket.total_size_bytes = 0;
        bucket.is_prefetching = false;
        
        // Fill bucket up to bucket size limit
        while (!prefetch_queue_.empty() && bucket.total_size_bytes < prefetch_bucket_size_) {
            size_t param_id = prefetch_queue_.front();
            prefetch_queue_.pop();
            
            bucket.param_ids.push_back(param_id);
            bucket.total_size_bytes += 1024; // Assume 1KB per param for now (simplified)
        }
        
        if (!bucket.param_ids.empty()) {
            prefetch_buckets_.push_back(std::move(bucket));
        }
    }
}

void AdvancedPrefetchSystem::execute_bucket_prefetch(const PrefetchBucket& bucket) {
    // Stub implementsation - simulate prefetch work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::cout << "[AdvancedPrefetchSystem] Prefetched bucket with " 
              << bucket.param_ids.size() << " parameters ("
              << bucket.total_size_bytes / 1024 << " KB)" << std::endl;
}

bool AdvancedPrefetchSystem::can_schedule_more_prefetches() const {
    return prefetch_buckets_.size() < max_prefetch_buckets_;
}

// ===============================
// AsyncIOManager Implementation
// ===============================

AsyncIOManager::AsyncIOManager(size_t num_io_threads) 
    : io_active_(false),  // Keep false during construction
      total_reads_(0),
      total_writes_(0), 
      total_bytes_read_(0),
      total_bytes_written_(0),
      io_errors_(0) {
    
    std::cout << "[AsyncIOManager] Started with " << num_io_threads << " IO threads" << std::endl;
    
    // Start IO worker threads AFTER object is fully constructed
    io_active_ = true;
    io_worker_threads_.reserve(num_io_threads);
    for (size_t i = 0; i < num_io_threads; ++i) {
        io_worker_threads_.emplace_back(&AsyncIOManager::io_worker_loop, this);
    }
}

AsyncIOManager::~AsyncIOManager() {
    io_active_ = false;
    io_cv_.notify_all();
    
    for (auto& thread : io_worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    std::cout << "[AsyncIOManager] Destroyed after " << total_reads_ + total_writes_ 
              << " IO operations, " << (total_bytes_read_ + total_bytes_written_) / (1024*1024) 
              << "MB transferred" << std::endl;
}

void AsyncIOManager::async_read(size_t param_id, const std::string& file_path, void* buffer, size_t size,
                               const std::function<void(bool)>& callback) {
    if (!io_active_) {
        if (callback) callback(false);
        return;
    }
    
    std::lock_guard<std::mutex> lock(io_mutex_);
    
    IORequest request;
    request.type = IORequest::READ;
    request.param_id = param_id;
    request.file_path = file_path;
    request.buffer = buffer;
    request.size = size;
    request.callback = callback;
    request.submit_time = std::chrono::steady_clock::now();
    
    io_queue_.push(std::move(request));
    io_cv_.notify_one();
}

void AsyncIOManager::async_write(size_t param_id, const std::string& file_path, const void* buffer, size_t size,
                                const std::function<void(bool)>& callback) {
    if (!io_active_) {
        if (callback) callback(false);
        return;
    }
    
    std::lock_guard<std::mutex> lock(io_mutex_);
    
    IORequest request;
    request.type = IORequest::WRITE;
    request.param_id = param_id;
    request.file_path = file_path;
    request.buffer = const_cast<void*>(buffer);
    request.size = size;
    request.callback = callback;
    request.submit_time = std::chrono::steady_clock::now();
    
    io_queue_.push(std::move(request));
    io_cv_.notify_one();
}

void AsyncIOManager::batch_read(const std::vector<size_t>& param_ids, 
                               const std::vector<std::string>& file_paths,
                               const std::vector<void*>& buffers, 
                               const std::vector<size_t>& sizes,
                               const std::function<void(size_t, bool)>& callback) {
    if (param_ids.size() != file_paths.size() || param_ids.size() != buffers.size() || 
        param_ids.size() != sizes.size()) {
        if (callback) {
            for (size_t i = 0; i < param_ids.size(); ++i) {
                callback(i, false);
            }
        }
        return;
    }
    
    for (size_t i = 0; i < param_ids.size(); ++i) {
        async_read(param_ids[i], file_paths[i], buffers[i], sizes[i], 
                  [callback, i](bool success) {
                      if (callback) callback(i, success);
                  });
    }
}

AsyncIOManager::IOStats AsyncIOManager::get_io_stats() const {
    IOStats stats;
    stats.total_reads = total_reads_.load();
    stats.total_writes = total_writes_.load();
    stats.total_bytes_read = total_bytes_read_.load();
    stats.total_bytes_written = total_bytes_written_.load();
    stats.io_errors = io_errors_.load();
    
    // Calculate average speeds (simplified)
    stats.average_read_speed_mbps = stats.total_bytes_read > 0 ? 
        static_cast<double>(stats.total_bytes_read) / (1024*1024) : 0.0;
    stats.average_write_speed_mbps = stats.total_bytes_written > 0 ?
        static_cast<double>(stats.total_bytes_written) / (1024*1024) : 0.0;
    
    return stats;
}

void AsyncIOManager::io_worker_loop() {
    while (io_active_) {
        std::unique_lock<std::mutex> lock(io_mutex_);
        io_cv_.wait(lock, [this] { return !io_queue_.empty() || !io_active_; });
        
        if (!io_active_) break;
        
        if (!io_queue_.empty()) {
            IORequest request = io_queue_.front();
            io_queue_.pop();
            lock.unlock();
            
            // Execute the IO request
            bool success = execute_io_request(request);
            
            // Update statistics
            if (request.type == IORequest::READ) {
                total_reads_++;
                if (success) {
                    total_bytes_read_ += request.size;
                } else {
                    io_errors_++;
                }
            } else if (request.type == IORequest::WRITE) {
                total_writes_++;
                if (success) {
                    total_bytes_written_ += request.size;
                } else {
                    io_errors_++;
                }
            }
            
            // Call callback if providesd
            if (request.callback) {
                request.callback(success);
            }
        }
    }
}

bool AsyncIOManager::execute_io_request(const IORequest& request) {
    // Stub implementsation - simulate IO operation
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Simulate IO delay
    
    if (request.type == IORequest::READ) {
        // Simulate reading from file
        std::ifstream file(request.file_path, std::ios::binary);
        if (file && request.buffer) {
            file.read(static_cast<char*>(request.buffer), request.size);
            return file.good() || file.eof();
        }
    } else if (request.type == IORequest::WRITE) {
        // Simulate writing to file
        std::ofstream file(request.file_path, std::ios::binary);
        if (file && request.buffer) {
            file.write(static_cast<const char*>(request.buffer), request.size);
            return file.good();
        }
    }
    
    return false; // Failed
}

} // namespace memory
} // namespace ops
