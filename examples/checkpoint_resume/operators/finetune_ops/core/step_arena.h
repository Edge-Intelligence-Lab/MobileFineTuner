/**
 * @file step_arena.h  
 * @brief Step-based memory arena allocator for temporary per-step memory
 * 
 * Provides a simple arena allocator that can be reset after each training step,
 * avoiding repeated malloc/free calls and reducing memory fragmentation.
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace ops {

class StepArena {
private:
    std::vector<char> buffer_;
    size_t offset_;
    size_t capacity_;
    
public:
    explicit StepArena(size_t capacity_mb = 64) 
        : offset_(0), capacity_(capacity_mb * 1024 * 1024) {
        buffer_.resize(capacity_);
    }
    
    /**
     * @brief Allocate memory from the arena with specified alignment
     */
    void* allocate(size_t size, size_t alignment = 64) {
        // Align offset to alignment boundary
        size_t aligned_offset = (offset_ + alignment - 1) / alignment * alignment;
        
        if (aligned_offset + size > capacity_) {
            throw std::runtime_error("StepArena exhausted: need " + std::to_string(size) + 
                                   " bytes but only " + std::to_string(capacity_ - aligned_offset) + " available");
        }
        
        void* ptr = &buffer_[aligned_offset];
        offset_ = aligned_offset + size;
        
        return ptr;
    }
    
    /**
     * @brief Allocate float array from the arena
     */
    float* allocate_floats(size_t count) {
        return static_cast<float*>(allocate(count * sizeof(float), 64));
    }
    
    /**
     * @brief Reset the arena (all previous allocations are invalidated)
     */
    void reset() {
        offset_ = 0;
    }
    
    /**
     * @brief Get current memory usage in bytes
     */
    size_t current_usage() const {
        return offset_;
    }
    
    size_t get_capacity() const {
        return capacity_;
    }
};

// Get global step arena instance
StepArena& get_step_arena();

} // namespace ops

