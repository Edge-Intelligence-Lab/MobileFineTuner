/**
 * @file mobile_safe_matmul.h
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * 
 * optimizationstrategy：
 * [Documentation available in English]
 * 2. adaptivechunkedoptimization - 5-10xperforanceboost  
 * [Documentation available in English]
 * [Documentation available in English]
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/task.h>
#endif

namespace mobile_matmul {

/**
 * [Documentation available in English]
 */
class MemorySafetyMonitor {
public:
    static size_t get_available_memory() {
        #ifdef __APPLE__
        struct task_basic_info info;
        mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
            return 8ULL * 1024 * 1024 * 1024 - info.resident_size;             // [Translated]
        }
        #endif
        return 2ULL * 1024 * 1024 * 1024;         // [Translated]
    }
    
    static size_t get_l1_cache_size() {
        #ifdef __APPLE__
        size_t cache_size = 0;
        size_t size = sizeof(cache_size);
        if (sysctlbyname("hw.l1dcachesize", &cache_size, &size, NULL, 0) == 0) {
            return cache_size;
        }
        #endif
        return 32 * 1024; // default32KB L1cache
    }
    
    static bool is_operation_safe(int64_t m, int64_t n, int64_t k) {
        size_t matrix_memory = (m*k + k*n + m*n) * sizeof(float);
        size_t available = get_available_memory();
        
                // [Translated]
        return matrix_memory < (available * 0.25);
    }
};

/**
 * [Documentation available in English]
 */
class AdaptiveBlockSize {
public:
    static int calculate_safe_block_size() {
        size_t l1_cache = MemorySafetyMonitor::get_l1_cache_size();
        
        // [Translated comment removed - see documentation]
        int max_block = static_cast<int>(std::sqrt(l1_cache / (6 * sizeof(float))));
        
                // [Translated]
        max_block = std::min(max_block, 128);
        max_block = std::max(max_block, 16);
        
        // [Translated comment removed - see documentation]
        return (max_block / 4) * 4;
    }
    
    static int calculate_block_size_for_matrix(int64_t m, int64_t n, int64_t k) {
        int base_block = calculate_safe_block_size();
        
        // according tomatrixsizedynamicadjust
        if (m < 256 && n < 256 && k < 256) {
            return std::min(base_block, 32); // smallmatrixusesmallblock
        } else if (m > 1024 || n > 1024 || k > 1024) {
            return std::min(base_block, 64); // largematrixcontrolblocksize
        }
        
        return base_block;
    }
};

/**
 * [Documentation available in English]
 */
class PerforanceMonitor {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    long get_elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time_).count();
    }
    
    double calculate_gflops(int64_t m, int64_t n, int64_t k, long time_ms) {
        if (time_ms == 0) return 0.0;
        double operations = 2.0 * m * n * k;         // [Translated]
        double time_sec = time_ms / 1000.0;
        return (operations / time_sec) / 1e9; // GFLOPS
    }
};

/**
 * [Documentation available in English]
 */
void naive_matmul(const float* A, const float* B, float* C,
                  int64_t M, int64_t N, int64_t K);

/**
 * [Documentation available in English]
 */
void reordered_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t N, int64_t K);

/**
 * @brief chunkedoptimizationmatrix multiplication
 */
void blocked_matmul(const float* A, const float* B, float* C,
                    int64_t M, int64_t N, int64_t K, int block_size = 0);

/**
 * @brief ARM NEONtoquantizationmatrix multiplication（ifsupport）
 */
void vectorized_matmul(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K);

/**
 * [Documentation available in English]
 */
void adaptive_matmul(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K);

/**
 * [Documentation available in English]
 */
enum class OptimizationLevel {
    NAIVE,          // [Translated]
    REORDERED,      // [Translated]
    BLOCKED,    // chunkedoptimization
    VECTORIZED, // SIMDtoquantization
    ADAPTIVE,   // adaptiveselect
    MEMORY_FIRST     // [Translated]
};

/**
 * @brief mainsafematrix multiplicationinterface
 */
class SafeMatmul {
public:
    static OptimizationLevel select_best_strategy(int64_t m, int64_t n, int64_t k);
    
    static void multiply(const float* A, const float* B, float* C,
                        int64_t M, int64_t N, int64_t K,
                        OptimizationLevel level = OptimizationLevel::ADAPTIVE);
    
    /**
     * @brief rightmatrixtransposematrix multiplication（used for tying，zero-copy）
     * [Documentation available in English]
     */
    static void multiply_rhs_T(const float* A, const float* B, float* C,
                              int64_t M, int64_t N, int64_t K,
                              OptimizationLevel level = OptimizationLevel::ADAPTIVE);
    
    static void benchmark_all_methods(int64_t M, int64_t N, int64_t K);
};

} // namespace mobile_matmul
