/**
 * @file mobile_param_manager.cpp
 * @brief Implementation of mobile-optimized parameter management system
 */

#include "mobile_param_manager.h"
#include "mobile_param_optimizations.h"
#include "mobile_specific_optimizations.h"
#include "../core/dtype.h"
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace ops {
namespace memory {

// ===============================
// ParameterCache Implementation
// ===============================

ParameterCache::ParameterCache(size_t max_size) 
    : max_size_(max_size), current_time_(0) {
    head_ = new CacheNode(0);
    tail_ = new CacheNode(0);
    head_->next = tail_;
    tail_->prev = head_;
}

ParameterCache::~ParameterCache() {
    clear();
    delete head_;
    delete tail_;
}

void ParameterCache::access(size_t param_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    current_time_++;
    
    auto it = cache_map_.find(param_id);
    if (it != cache_map_.end()) {
        // Cache hit - update access info and move to head
        CacheNode* node = it->second;
        node->last_access_time = current_time_;
        node->access_count++;
        move_to_head(node);
    } else if (cache_map_.size() < max_size_) {
        // Cache miss - add new node
        CacheNode* new_node = new CacheNode(param_id);
        new_node->last_access_time = current_time_;
        new_node->access_count = 1;
        cache_map_[param_id] = new_node;
        
        new_node->next = head_->next;
        new_node->prev = head_;
        head_->next->prev = new_node;
        head_->next = new_node;
    }
}

std::vector<size_t> ParameterCache::get_eviction_candidates(size_t count) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::vector<size_t> candidates;
    
    CacheNode* current = tail_->prev;
    while (current != head_ && candidates.size() < count) {
        candidates.push_back(current->param_id);
        current = current->prev;
    }
    
    return candidates;
}

void ParameterCache::remove(size_t param_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_map_.find(param_id);
    if (it != cache_map_.end()) {
        remove_node(it->second);
        delete it->second;
        cache_map_.erase(it);
    }
}

void ParameterCache::move_to_head(CacheNode* node) {
    remove_node(node);
    node->next = head_->next;
    node->prev = head_;
    head_->next->prev = node;
    head_->next = node;
}

void ParameterCache::remove_node(CacheNode* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
}

double ParameterCache::get_hit_rate() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (current_time_ == 0) return 0.0;
    size_t total_hits = 0;
    for (const auto& pair : cache_map_) {
        total_hits += pair.second->access_count;
    }
    return static_cast<double>(total_hits) / current_time_;
}

void ParameterCache::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& pair : cache_map_) {
        delete pair.second;
    }
    cache_map_.clear();
    head_->next = tail_;
    tail_->prev = head_;
}

// ===============================
// AsyncParameterHandler Implementation
// ===============================

AsyncParameterHandler::AsyncParameterHandler() : stop_flag_(false) {
    worker_thread_ = std::thread(&AsyncParameterHandler::worker_loop, this);
}

AsyncParameterHandler::~AsyncParameterHandler() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_flag_ = true;
    }
    queue_cv_.notify_one();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void AsyncParameterHandler::schedule_load(size_t param_id, const std::function<void()>& load_fn) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Use param_id for logging to avoid unused parameter warning
        (void)param_id;  // Suppress warning - could be used for priority scheduling
        task_queue_.push(load_fn);
    }
    queue_cv_.notify_one();
}

void AsyncParameterHandler::schedule_unload(size_t param_id, const std::function<void()>& unload_fn) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Use param_id for logging to avoid unused parameter warning
        (void)param_id;  // Suppress warning - could be used for priority scheduling
        task_queue_.push(unload_fn);
    }
    queue_cv_.notify_one();
}

void AsyncParameterHandler::wait_for_completion() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return task_queue_.empty(); });
}

void AsyncParameterHandler::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || stop_flag_; });
            
            if (stop_flag_ && task_queue_.empty()) {
                break;
            }
            
            task = task_queue_.front();
            task_queue_.pop();
        }
        
        try {
            task();
        } catch (const std::exception& e) {
            // Log error but continue processing
            std::cerr << "Error in async parameter handler: " << e.what() << std::endl;
        }
    }
}

// ===============================
// MobileParameterManager Implementation
// ===============================

MobileParameterManager::MobileParameterManager(const MobileParamConfig& config) 
    : config_(config), 
      gpu_memory_pool_(nullptr), 
      cpu_memory_pool_(nullptr),
      gpu_memory_offset_(0), 
      cpu_memory_offset_(0),
      prefetch_enabled_(config.enable_prefetch),
      mobile_optimization_active_(false),
      mobile_optimization_effectiveness_(0.0) {
    
    try {
        std::cout << "[MobileParameterManager] Starting safe initialization..." << std::endl;
        
        // Initialize basic components first
        param_cache_ = std::make_unique<ParameterCache>(1000);
        async_handler_ = std::make_unique<AsyncParameterHandler>();
        
        // Initialize optimization components with null checks
        std::cout << "[MobileParameterManager] Initializing optimization components..." << std::endl;
        
        if (config_.enable_quantization) {
            parameter_quantizer_ = std::make_unique<ParameterQuantizer>(config_.default_quantization);
            if (parameter_quantizer_) {
                std::cout << "  Parameter quantization initialized" << std::endl;
            }
        }
        
        if (config_.enable_pin_memory) {
            pinned_memory_manager_ = std::make_unique<PinnedMemoryManager>(config_.max_pinned_memory_mb);
            if (pinned_memory_manager_) {
                std::cout << "  Pin memory manager initialized" << std::endl;
            }
        }
        
        contiguous_allocator_ = std::make_unique<ContiguousMemoryAllocator>(
            config_.max_gpu_memory_mb * 1024 * 1024,  // GPU pool size
            config_.memory_alignment
        );
        if (contiguous_allocator_) {
            std::cout << "  Contiguous memory allocator initialized" << std::endl;
        }
        
        persistence_manager_ = std::make_unique<ParameterPersistenceManager>(
            config_.param_persistence_threshold,
            config_.model_persistence_threshold
        );
        if (persistence_manager_) {
            std::cout << "  Parameter persistence manager initialized" << std::endl;
        }
        
        if (config_.enable_reuse_tracking) {
            reuse_tracker_ = std::make_unique<ParameterReuseTracker>(
                config_.max_reuse_distance
            );
            if (reuse_tracker_) {
                std::cout << "  Parameter reuse tracker initialized" << std::endl;
            }
        }
        
        if (config_.enable_thermal_monitoring) {
            platfor_optimizer_ = std::make_unique<MobilePlatforOptimizer>();
            if (platfor_optimizer_) {
                std::cout << "  Mobile platfor optimizer initialized" << std::endl;
            }
        }
        
        if (config_.enable_predictive_prefetch) {
            advanced_prefetch_ = std::make_unique<AdvancedPrefetchSystem>(
                config_.prefetch_bucket_size
            );
            if (advanced_prefetch_) {
                std::cout << "  Advanced prefetch system initialized" << std::endl;
            }
        }
        
        if (config_.async_io_threads > 0) {
            async_io_manager_ = std::make_unique<AsyncIOManager>(
                config_.async_io_threads
            );
            if (async_io_manager_) {
                std::cout << "  Async I/O manager initialized" << std::endl;
            }
        }
        
        // Initialize mobile optimization coordinator
        if (config_.enable_mobile_optimization_coordinator) {
            mobile_coordinator_ = std::make_unique<::ops::memory::MobileOptimizationCoordinator>();
            if (mobile_coordinator_) {
                mobile_optimization_active_ = true;
                mobile_optimization_effectiveness_ = 0.0;
                std::cout << "  Mobile optimization coordinator initialized" << std::endl;
            }
        }
        
        // Allocate memory pools
        allocate_memory_pools();
        
        // Initialize statistics
        memory_stats_ = {};
        
        // Create storage directory
        if (config_.enable_storage_offload) {
            std::filesystem::create_directories(config_.storage_path);
        }
        
        // Start prefetch thread AFTER all initialization is complete
        if (prefetch_enabled_) {
            prefetch_thread_ = std::thread(&MobileParameterManager::prefetch_worker_loop, this);
        }
        
        std::cout << "[MobileParameterManager] Initialized successfully with GPU:" << config_.max_gpu_memory_mb 
                  << "MB, CPU:" << config_.max_cpu_memory_mb << "MB" << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "[MobileParameterManager] CRITICAL ERROR during initialization: " << e.what() << std::endl;
        // Clean up any partially initialized state
        prefetch_enabled_ = false;
        mobile_optimization_active_ = false;
        throw; // Re-throw to indicate construction failure
    }
}

MobileParameterManager::~MobileParameterManager() {
    // Stop prefetch thread
    prefetch_enabled_ = false;
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    
    // Clean up memory pools
    cleanup_memory_pools();
}

size_t MobileParameterManager::register_parameter(const std::string& name, const TensorPtr& tensor, bool is_trainable) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    size_t param_id = partitions_.size();
    size_t param_size = tensor->numel() * DTypeUtils::size_of(tensor->dtype());
    
    // Use is_trainable parameter for optimization decisions
    (void)is_trainable;
    
    auto partition = std::make_unique<ParameterPartition>(param_id, 0, param_size);
    partition->tensor = tensor;
    
    // Determine initial storage tier based on size and available memory
    if (param_size <= config_.max_gpu_memory_mb * 1024 * 1024 / 4) {  // Use 1/4 of GPU memory for large parameters
        partition->current_tier = MemoryTier::GPU_MEMORY;
    } else {
        partition->current_tier = MemoryTier::CPU_MEMORY;
        if (config_.enable_storage_offload && param_size > config_.max_cpu_memory_mb * 1024 * 1024 / 8) {
            partition->current_tier = MemoryTier::STORAGE;
            partition->status = ParameterStatus::OFFLOADED;
        }
    }
    
    partitions_.push_back(std::move(partition));
    param_name_to_id_[name] = param_id;
    
    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        memory_stats_.total_params_size += param_size;
    }
    
    std::cout << "[MobileParamMgr] Registered parameter '" << name << "' (ID=" << param_id 
              << ", Size=" << param_size / 1024 / 1024 << "MB, Tier=" << static_cast<int>(partitions_[param_id]->current_tier) 
              << ")" << std::endl;
    
    return param_id;
}

TensorPtr MobileParameterManager::get_parameter(size_t param_id, const std::string& hint) {
    if (param_id >= partitions_.size()) {
        throw std::invalid_argument("Invalid parameter ID: " + std::to_string(param_id));
    }
    
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto& partition = partitions_[param_id];
    
    // Use hint for optimization (currently unused but reserved for future use)
    (void)hint;
    
    // Update cache access pattern
    param_cache_->access(param_id);
    update_access_pattern(param_id);
    
    // Ensure parameter is loaded
    if (partition->status != ParameterStatus::AVAILABLE) {
        load_parameter_sync(param_id);
    }
    
    // Trigger eviction if memory pressure is high
    trigger_eviction_if_needed();
    
    return partition->tensor;
}

TensorPtr MobileParameterManager::get_parameter(const std::string& name, const std::string& hint) {
    auto it = param_name_to_id_.find(name);
    if (it == param_name_to_id_.end()) {
        throw std::invalid_argument("Parameter not found: " + name);
    }
    return get_parameter(it->second, hint);
}

void MobileParameterManager::release_parameter(size_t param_id, bool mark_dirty) {
    if (param_id >= partitions_.size()) return;
    
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto& partition = partitions_[param_id];
    
    if (mark_dirty && partition->current_tier == MemoryTier::STORAGE) {
        // If parameter was modified and is in storage, save it back
        save_parameter_to_storage(param_id);
    }
}

void MobileParameterManager::prefetch_parameters(const std::vector<size_t>& param_ids, MemoryTier tier) {
    if (!config_.enable_prefetch) return;
    
    // Use memory tier for optimization decisions
    (void)tier;
    
    for (size_t param_id : param_ids) {
        if (param_id < partitions_.size()) {
            async_handler_->schedule_load(param_id, [this, param_id]() {
                std::lock_guard<std::mutex> lock(manager_mutex_);
                load_parameter_sync(param_id);
            });
        }
    }
}

void MobileParameterManager::evict_parameters(size_t bytes_needed, MemoryTier preferred_tier) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    // Get eviction candidates from cache
    auto candidates = param_cache_->get_eviction_candidates(10);
    
    size_t bytes_freed = 0;
    for (size_t param_id : candidates) {
        if (bytes_freed >= bytes_needed) break;
        
        auto& partition = partitions_[param_id];
        if (partition->current_tier == preferred_tier && partition->status == ParameterStatus::AVAILABLE) {
            // Move to next tier
            MemoryTier next_tier = (preferred_tier == MemoryTier::GPU_MEMORY) ? 
                                  MemoryTier::CPU_MEMORY : MemoryTier::STORAGE;
            
            if (next_tier == MemoryTier::STORAGE) {
                save_parameter_to_storage(param_id);
            } else {
                // Move from GPU to CPU memory
                // Implementation would copy tensor data
            }
            
            bytes_freed += partition->size_bytes;
            param_cache_->remove(param_id);
            
            std::cout << "[MobileParamMgr] Evicted parameter " << param_id 
                      << " (" << partition->size_bytes / 1024 / 1024 << "MB)" << std::endl;
        }
    }
}

void MobileParameterManager::save_checkpoint(const std::string& checkpoint_path) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    std::filesystem::create_directories(std::filesystem::path(checkpoint_path).parent_path());
    std::ofstream checkpoint_file(checkpoint_path, std::ios::binary);
    
    if (!checkpoint_file) {
        throw std::runtime_error("Cannot create checkpoint file: " + checkpoint_path);
    }
    
    // Write header
    size_t num_params = partitions_.size();
    checkpoint_file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    
    // Write each parameter
    for (const auto& partition : partitions_) {
        // Ensure parameter is loaded
        if (partition->status != ParameterStatus::AVAILABLE) {
            load_parameter_sync(partition->partition_id);
        }
        
        // Write parameter metadata
        checkpoint_file.write(reinterpret_cast<const char*>(&partition->partition_id), sizeof(partition->partition_id));
        checkpoint_file.write(reinterpret_cast<const char*>(&partition->size_bytes), sizeof(partition->size_bytes));
        
        // Write parameter data
        const void* data = partition->tensor->data_ptr();
        checkpoint_file.write(reinterpret_cast<const char*>(data), partition->size_bytes);
    }
    
    checkpoint_file.close();
    std::cout << "[MobileParamMgr] Saved checkpoint with " << num_params << " parameters" << std::endl;
}

void MobileParameterManager::load_checkpoint(const std::string& checkpoint_path) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    std::ifstream checkpoint_file(checkpoint_path, std::ios::binary);
    if (!checkpoint_file) {
        throw std::runtime_error("Cannot open checkpoint file: " + checkpoint_path);
    }
    
    // Read header
    size_t num_params;
    checkpoint_file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    
    // Read each parameter
    for (size_t i = 0; i < num_params; ++i) {
        size_t param_id, size_bytes;
        checkpoint_file.read(reinterpret_cast<char*>(&param_id), sizeof(param_id));
        checkpoint_file.read(reinterpret_cast<char*>(&size_bytes), sizeof(size_bytes));
        
        if (param_id < partitions_.size()) {
            auto& partition = partitions_[param_id];
            
            // Allocate memory if needed
            if (!partition->tensor || partition->tensor->numel() == 0) {
                // Create new tensor - implementsation depends on your tensor API
                // partition->tensor = create_tensor_from_size(size_bytes);
            }
            
            // Read parameter data
            void* data = const_cast<void*>(partition->tensor->data_ptr());
            checkpoint_file.read(reinterpret_cast<char*>(data), size_bytes);
            
            partition->status = ParameterStatus::AVAILABLE;
            partition->current_tier = MemoryTier::GPU_MEMORY; // Load to fastest memory
        }
    }
    
    checkpoint_file.close();
    std::cout << "[MobileParamMgr] Loaded checkpoint with " << num_params << " parameters" << std::endl;
}

MemoryStats MobileParameterManager::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return memory_stats_;
}

void MobileParameterManager::load_parameter_sync(size_t param_id) {
    auto& partition = partitions_[param_id];
    
    if (partition->status == ParameterStatus::AVAILABLE) {
        return; // Already loaded
    }
    
    partition->status = ParameterStatus::LOADING;
    
    switch (partition->current_tier) {
        case MemoryTier::STORAGE:
            load_parameter_from_storage(param_id);
            break;
        case MemoryTier::CPU_MEMORY:
            // Move from CPU to GPU memory if needed
            // Implementation depends on your tensor API
            break;
        default:
            break;
    }
    
    partition->status = ParameterStatus::AVAILABLE;
}

void MobileParameterManager::save_parameter_to_storage(size_t param_id) {
    auto& partition = partitions_[param_id];
    
    std::string filename = config_.storage_path + "/param_" + std::to_string(param_id) + ".bin";
    std::ofstream file(filename, std::ios::binary);
    
    if (file) {
        const void* data = partition->tensor->data_ptr();
        file.write(reinterpret_cast<const char*>(data), partition->size_bytes);
        partition->storage_path = filename;
        partition->current_tier = MemoryTier::STORAGE;
        partition->status = ParameterStatus::OFFLOADED;
    }
}

void MobileParameterManager::load_parameter_from_storage(size_t param_id) {
    auto& partition = partitions_[param_id];
    
    if (partition->storage_path.empty()) {
        partition->storage_path = config_.storage_path + "/param_" + std::to_string(param_id) + ".bin";
    }
    
    std::ifstream file(partition->storage_path, std::ios::binary);
    if (file) {
        void* data = const_cast<void*>(partition->tensor->data_ptr());
        file.read(reinterpret_cast<char*>(data), partition->size_bytes);
        partition->current_tier = MemoryTier::CPU_MEMORY; // Load to CPU first
    }
}

void MobileParameterManager::allocate_memory_pools() {
    // Allocate GPU memory pool
    size_t gpu_pool_size = config_.max_gpu_memory_mb * 1024 * 1024;
    gpu_memory_pool_ = std::aligned_alloc(64, gpu_pool_size); // 64-byte aligned
    
    // Allocate CPU memory pool  
    size_t cpu_pool_size = config_.max_cpu_memory_mb * 1024 * 1024;
    cpu_memory_pool_ = std::aligned_alloc(64, cpu_pool_size);
    
    if (!gpu_memory_pool_ || !cpu_memory_pool_) {
        throw std::runtime_error("Failed to allocate memory pools");
    }
    
    gpu_memory_offset_ = 0;
    cpu_memory_offset_ = 0;
}

void MobileParameterManager::cleanup_memory_pools() {
    if (gpu_memory_pool_) {
        std::free(gpu_memory_pool_);
        gpu_memory_pool_ = nullptr;
    }
    
    if (cpu_memory_pool_) {
        std::free(cpu_memory_pool_);
        cpu_memory_pool_ = nullptr;
    }
}

void MobileParameterManager::update_access_pattern(size_t param_id) {
    // Update statistics
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    memory_stats_.total_alloc_requests++;
    
    // Use param_id for access pattern analysis
    (void)param_id;
    
    // Simple access pattern tracking - could be enhanced
    if (param_cache_) {
        double hit_rate = param_cache_->get_hit_rate();
        if (hit_rate > 0.8) {
            memory_stats_.cache_hit_count++;
        } else {
            memory_stats_.cache_miss_count++;
        }
    }
}

void MobileParameterManager::trigger_eviction_if_needed() {
    double memory_pressure = calculate_memory_pressure();
    
    if (memory_pressure > config_.eviction_threshold) {
        size_t bytes_to_free = static_cast<size_t>(
            (memory_pressure - config_.eviction_threshold) * config_.max_gpu_memory_mb * 1024 * 1024
        );
        evict_parameters(bytes_to_free, MemoryTier::GPU_MEMORY);
    }
}

double MobileParameterManager::calculate_memory_pressure() const {
    // Simple memory pressure calculation
    size_t total_available = config_.max_gpu_memory_mb * 1024 * 1024;
    size_t currently_used = gpu_memory_offset_;
    
    return static_cast<double>(currently_used) / total_available;
}

void MobileParameterManager::prefetch_worker_loop() {
    while (prefetch_enabled_) {
        // Simple prefetching logic - could be enhanced with ML-based prediction
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (!prefetch_queue_.empty()) {
            std::lock_guard<std::mutex> lock(manager_mutex_);
            
            auto it = prefetch_queue_.begin();
            if (it != prefetch_queue_.end()) {
                size_t param_id = *it;
                prefetch_queue_.erase(it);
                
                if (param_id < partitions_.size()) {
                    load_parameter_sync(param_id);
                }
            }
        }
    }
}

// ===============================
// ScopedParameterAccess Implementation
// ===============================

ScopedParameterAccess::ScopedParameterAccess(MobileParameterManager* manager, size_t param_id, const std::string& hint)
    : manager_(manager), param_id_(param_id), is_dirty_(false) {
    tensor_ = manager_->get_parameter(param_id, hint);
}

ScopedParameterAccess::ScopedParameterAccess(MobileParameterManager* manager, const std::string& name, const std::string& hint)
    : manager_(manager), param_id_(0), is_dirty_(false) {
    tensor_ = manager_->get_parameter(name, hint);
    // Would need to store param_id for proper cleanup
}

ScopedParameterAccess::~ScopedParameterAccess() {
    if (manager_ && param_id_ != SIZE_MAX) {
        manager_->release_parameter(param_id_, is_dirty_);
    }
}

ScopedParameterAccess::ScopedParameterAccess(ScopedParameterAccess&& other) noexcept
    : manager_(other.manager_), param_id_(other.param_id_), tensor_(std::move(other.tensor_)), is_dirty_(other.is_dirty_) {
    other.manager_ = nullptr;
    other.param_id_ = SIZE_MAX;
}

ScopedParameterAccess& ScopedParameterAccess::operator=(ScopedParameterAccess&& other) noexcept {
    if (this != &other) {
        // Clean up current resource
        if (manager_ && param_id_ != SIZE_MAX) {
            manager_->release_parameter(param_id_, is_dirty_);
        }
        
        // Move from other
        manager_ = other.manager_;
        param_id_ = other.param_id_;
        tensor_ = std::move(other.tensor_);
        is_dirty_ = other.is_dirty_;
        
        other.manager_ = nullptr;
        other.param_id_ = SIZE_MAX;
    }
    return *this;
}

// ===============================
// Factory Function
// ===============================

std::unique_ptr<MobileParameterManager> create_mobile_param_manager(
    size_t available_memory_mb,
    bool enable_storage_offload,
    const std::string& cache_dir
) {
    MobileParamConfig config;
    
    // Configure based on available memory
    config.max_gpu_memory_mb = available_memory_mb / 2;  // Use half for GPU simulation
    config.max_cpu_memory_mb = available_memory_mb;
    config.enable_storage_offload = enable_storage_offload;
    config.storage_path = cache_dir;
    
    // Adjust partition size based on available memory
    if (available_memory_mb < 512) {
        config.partition_size_mb = 16;   // Smaller partitions for low memory
        config.prefetch_buffer_mb = 32;
    } else if (available_memory_mb < 2048) {
        config.partition_size_mb = 32;
        config.prefetch_buffer_mb = 64;
    }
    
    return std::make_unique<MobileParameterManager>(config);
}

// ===============================
// Add missing method implementsations
// ===============================

float MobileParameterManager::get_current_temperature() const {
    // Since get_cpu_temperature() is private, we implements temperature detection directly
    // In a real implementsation, this would read from system sensors
    
    // Simulate temperature based on system activity (stub implementsation)
    static float base_temperature = 25.0f;
    static float temperature_variation = 0.1f;
    
    // Simple temperature simulation
    base_temperature += temperature_variation;
    if (base_temperature > 85.0f) {
        temperature_variation = -0.2f; // Cooling down
    } else if (base_temperature < 20.0f) {
        temperature_variation = 0.1f; // Warming up
    }
    
    return std::max(20.0f, std::min(90.0f, base_temperature));
}

const MobileParamConfig& MobileParameterManager::get_config() const {
    return config_;
}

void MobileParameterManager::deallocate_from_pool(MemoryTier tier, void* ptr, size_t size) {
    if (!ptr) return;
    
    // Use correct method signatures and enum values
    switch (tier) {
        case MemoryTier::GPU_MEMORY:
            if (contiguous_allocator_) {
                contiguous_allocator_->deallocate(ptr);
            }
            break;
            
        case MemoryTier::CPU_MEMORY:
            if (pinned_memory_manager_) {
                // free_pinned only takes one argument (ptr)
                pinned_memory_manager_->free_pinned(ptr);
            } else {
                // Fallback to standard free
                std::free(ptr);
            }
            break;
            
        case MemoryTier::STORAGE:
            // Use correct enum value STORAGE instead of PERSISTENT_STORAGE
            // Storage deallocations are handled by file operations
            // No direct memory deallocation needed
            (void)size; // Suppress unused parameter warning
            break;
            
        default:
            std::cerr << "[MobileParameterManager] Unknown memory tier for deallocation" << std::endl;
            (void)size; // Suppress unused parameter warning
            break;
    }
}

} // namespace memory
} // namespace ops
