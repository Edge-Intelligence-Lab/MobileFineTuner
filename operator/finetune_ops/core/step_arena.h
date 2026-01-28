/**
 * @file step_arena.h  
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
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
     * [Documentation available in English]
     */
    void* allocate(size_t size, size_t alignment = 64) {
                // [Translated]
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
     * [Documentation available in English]
     */
    float* allocate_floats(size_t count) {
        return static_cast<float*>(allocate(count * sizeof(float), 64));
    }
    
    /**
     * [Documentation available in English]
     */
    void reset() {
        offset_ = 0;
    }
    
    /**
     * [Documentation available in English]
     */
    size_t current_usage() const {
        return offset_;
    }
    
    size_t get_capacity() const {
        return capacity_;
    }
};

// [Translated]
StepArena& get_step_arena();

} // namespace ops

