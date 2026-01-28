/**
 * @file arena_allocator.h
 * @brief 分区内存管理系统 - 根治物理足迹线性增长
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

#ifdef __APPLE__
#include <sys/mman.h>
#elif defined(__linux__)
#include <sys/mman.h>
#endif

namespace ops {
namespace memory {

class StepScratchArena {
public:
    void* base_ptr_ = nullptr;  // Public for ArenaManager access

private:
    size_t capacity_ = 0;
    size_t offset_ = 0;
    size_t peak_usage_ = 0;
    size_t num_allocations_ = 0;
    size_t num_resets_ = 0;

    static constexpr size_t ALIGNMENT = 64;

public:
    explicit StepScratchArena(size_t capacity_mb = 128);
    ~StepScratchArena();

    void* allocate(size_t size);
    void reset();
    void recreate();  // 🔥 Generational arena: rebuild entirely, reset virtual space

    size_t current_usage() const { return offset_; }
    size_t peak_usage() const { return peak_usage_; }
    size_t capacity() const { return capacity_; }
    void print_stats() const;

    StepScratchArena(const StepScratchArena&) = delete;
    StepScratchArena& operator=(const StepScratchArena&) = delete;
};

class StaticWeightArena {
private:
    struct WeightBlock {
        void* ptr = nullptr;
        size_t size = 0;
        std::string name;
    };

    std::vector<WeightBlock> blocks_;
    size_t total_size_ = 0;
    mutable std::mutex mutex_;

public:
    StaticWeightArena() = default;
    ~StaticWeightArena();

    void* allocate_static(size_t size, const std::string& name = "");

    size_t total_size() const { return total_size_; }
    void print_stats() const;

    StaticWeightArena(const StaticWeightArena&) = delete;
    StaticWeightArena& operator=(const StaticWeightArena&) = delete;
};

class DirectLargeAllocator {
private:
    struct LargeBlock {
        void* ptr = nullptr;
        size_t size = 0;
    };

    std::unordered_map<void*, LargeBlock> allocations_;
    size_t total_allocated_ = 0;
    size_t num_allocations_ = 0;
    mutable std::mutex mutex_;

    static constexpr size_t LARGE_THRESHOLD = 16 * 1024 * 1024;  // 16MB

public:
    DirectLargeAllocator() = default;
    ~DirectLargeAllocator();

    static bool is_large(size_t size) { return size >= LARGE_THRESHOLD; }

    void* allocate(size_t size);
    void free(void* ptr);

    size_t total_allocated() const { return total_allocated_; }
    void print_stats() const;
};

class ArenaManager {
private:
    std::unique_ptr<StaticWeightArena> static_arena_;
    std::unique_ptr<DirectLargeAllocator> large_allocator_;

    static thread_local StepScratchArena* current_step_arena_;

    mutable std::mutex mutex_;

    ArenaManager();

public:
    ~ArenaManager();

    static ArenaManager& instance();

    void set_current_step_arena(StepScratchArena* arena);
    StepScratchArena* get_current_step_arena();
    void clear_current_step_arena();

    StaticWeightArena& static_weights() { return *static_arena_; }
    DirectLargeAllocator& large_alloc() { return *large_allocator_; }

    void* allocate(size_t size);
    void free(void* ptr, size_t size);

    void print_all_stats() const;

    ArenaManager(const ArenaManager&) = delete;
    ArenaManager& operator=(const ArenaManager&) = delete;
};

class StepArenaGuard {
private:
    StepScratchArena arena_;

public:
    explicit StepArenaGuard(size_t capacity_mb = 128)
        : arena_(capacity_mb) {
        ArenaManager::instance().set_current_step_arena(&arena_);
    }

    ~StepArenaGuard() {
        ArenaManager::instance().clear_current_step_arena();
        arena_.reset();
    }

    StepScratchArena& get_arena() { return arena_; }
    
    // 🔥 Generational arena: force rebuild to prevent macOS footprint growth
    void regenerate() {
        arena_.recreate();
    }

    StepArenaGuard(const StepArenaGuard&) = delete;
    StepArenaGuard& operator=(const StepArenaGuard&) = delete;
};

} // namespace memory
} // namespace ops


