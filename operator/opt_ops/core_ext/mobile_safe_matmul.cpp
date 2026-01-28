/**
 * @file mobile_safe_matmul.cpp
 * [Documentation available in English]
 */

#include "mobile_safe_matmul.h"
#include <cstring>
#include <vector>
#include <iomanip>

namespace mobile_matmul {

/**
 * [Documentation available in English]
 */
void naive_matmul(const float* A, const float* B, float* C,
                  int64_t M, int64_t N, int64_t K) {
    // initializeresultmatrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // standardijkloopsequential
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * [Documentation in English - see separate docs]
 * [Documentation available in English]
 */
void reordered_matmul(const float* A, const float* B, float* C,
                      int64_t M, int64_t N, int64_t K) {
    // initializeresultmatrix
    std::memset(C, 0, M * N * sizeof(float));
    
        // [Translated]
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];              // [Translated]
            
                        // [Translated]
            for (int64_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/**
 * @brief chunkedoptimizationmatrix multiplication
 * 
 * [Documentation available in English]
 * [Documentation available in English]
 * [Documentation in English - see separate docs]
 * [Documentation in English - see separate docs]
 */
void blocked_matmul(const float* A, const float* B, float* C,
                    int64_t M, int64_t N, int64_t K, int block_size) {
    // [Translated comment removed - see documentation]
    if (block_size == 0) {
        block_size = AdaptiveBlockSize::calculate_block_size_for_matrix(M, N, K);
    }
    
    // initializeresultmatrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // chunkedCompute
    for (int64_t bi = 0; bi < M; bi += block_size) {
        for (int64_t bj = 0; bj < N; bj += block_size) {
            for (int64_t bk = 0; bk < K; bk += block_size) {
                
                // Computeactualblockboundary
                int64_t end_i = std::min(bi + block_size, M);
                int64_t end_j = std::min(bj + block_size, N);
                int64_t end_k = std::min(bk + block_size, K);
                
                // atblockinneruseoptimizationikjloop
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief ARM NEONtoquantizationmatrix multiplication
 */
void vectorized_matmul(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K) {
#ifdef __ARM_NEON
    // initializeresultmatrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // useNEONtoquantizationikjloop
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float32x4_t a_ik_vec = vdupq_n_f32(A[i * K + k]);
            
            int64_t j = 0;
            // [Translated comment removed - see documentation]
            for (; j <= N - 4; j += 4) {
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                float32x4_t c_vec = vld1q_f32(&C[i * N + j]);
                
                // C[i][j:j+4] += A[i][k] * B[k][j:j+4]
                c_vec = vmlaq_f32(c_vec, a_ik_vec, b_vec);
                
                vst1q_f32(&C[i * N + j], c_vec);
            }
            
            // [Translated comment removed - see documentation]
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
#else
        // [Translated]
    reordered_matmul(A, B, C, M, N, K);
#endif
}

/**
 * [Documentation available in English]
 */
OptimizationLevel SafeMatmul::select_best_strategy(int64_t m, int64_t n, int64_t k) {
        // [Translated]
    if (!MemorySafetyMonitor::is_operation_safe(m, n, k)) {
        std::cout << "⚠️ matrixthroughlarge，[Output]atmemory[Output]，usefundamentalimplements" << std::endl;
        return OptimizationLevel::NAIVE;
    }
    
        // [Translated]
    if (m < 64 && n < 64 && k < 64) {
        return OptimizationLevel::REORDERED;
    }
    
        // [Translated]
    if (m < 512 && n < 512 && k < 512) {
        return OptimizationLevel::BLOCKED;
    }
    
    // largematrixusetoquantization+chunked
    #ifdef __ARM_NEON
    return OptimizationLevel::VECTORIZED;
    #else
    return OptimizationLevel::BLOCKED;
    #endif
}

/**
 * [Documentation available in English]
 * [Documentation available in English]
 */
void memory_first_matmul(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K) {
    // initializeresultmatrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // [Translated comment removed - see documentation]
    const int64_t block_size = 16;
    
        // [Translated]
    for (int64_t bi = 0; bi < M; bi += block_size) {
        int64_t end_i = std::min(bi + block_size, M);
        
        for (int64_t bj = 0; bj < N; bj += block_size) {
            int64_t end_j = std::min(bj + block_size, N);
            
            for (int64_t bk = 0; bk < K; bk += block_size) {
                int64_t end_k = std::min(bk + block_size, K);
                
                                // [Translated]
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                                                // [Translated]
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief mainsafematrix multiplicationinterface
 */
void SafeMatmul::multiply(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K,
                         OptimizationLevel level) {
    // adaptiveselectstrategy
    if (level == OptimizationLevel::ADAPTIVE) {
        level = select_best_strategy(M, N, K);
    }
    
    // according toselectstrategyexecutematrix multiplication
    switch (level) {
        case OptimizationLevel::NAIVE:
            naive_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::REORDERED:
            reordered_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::BLOCKED:
            blocked_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::VECTORIZED:
            vectorized_matmul(A, B, C, M, N, K);
            break;
            
        case OptimizationLevel::MEMORY_FIRST:
            memory_first_matmul(A, B, C, M, N, K);
            break;
            
        default:
            reordered_matmul(A, B, C, M, N, K);
            break;
    }
}

/**
 * @brief rightmatrixtransposematrix multiplication（MEMORY_FIRST version，zero-copy）
 * [Documentation available in English]
 */
void memory_first_matmul_rhs_T(const float* A, const float* B, float* C,
                               int64_t M, int64_t N, int64_t K) {
    std::memset(C, 0, M * N * sizeof(float));
    
    const int64_t block_size = 16;
    
    // chunkedCompute（M、N、K dimension）
    for (int64_t bi = 0; bi < M; bi += block_size) {
        int64_t end_i = std::min(bi + block_size, M);
        
        for (int64_t bj = 0; bj < N; bj += block_size) {
            int64_t end_j = std::min(bj + block_size, N);
            
            for (int64_t bk = 0; bk < K; bk += block_size) {
                int64_t end_k = std::min(bk + block_size, K);
                
                // blockinner ikj sequential
                for (int64_t i = bi; i < end_i; ++i) {
                    for (int64_t k = bk; k < end_k; ++k) {
                        float a_ik = A[i * K + k];
                        
                                                // [Translated]
                                                // [Translated]
                        for (int64_t j = bj; j < end_j; ++j) {
                            C[i * N + j] += a_ik * B[j * K + k];
                        }
                    }
                }
            }
        }
    }
}

/**
 * [Documentation available in English]
 */
void SafeMatmul::multiply_rhs_T(const float* A, const float* B, float* C,
                                int64_t M, int64_t N, int64_t K,
                                OptimizationLevel level) {
    // [Translated comment removed - see documentation]
    if (level == OptimizationLevel::MEMORY_FIRST || level == OptimizationLevel::ADAPTIVE) {
        memory_first_matmul_rhs_T(A, B, C, M, N, K);
    } else {
                // [Translated]
        memory_first_matmul_rhs_T(A, B, C, M, N, K);
    }
}

/**
 * [Documentation available in English]
 */
void adaptive_matmul(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K) {
    SafeMatmul::multiply(A, B, C, M, N, K, OptimizationLevel::ADAPTIVE);
}

/**
 * [Documentation available in English]
 */
void SafeMatmul::benchmark_all_methods(int64_t M, int64_t N, int64_t K) {
    std::cout << "\n=== matrix multiplicationperforance[Output]test ===" << std::endl;
    std::cout << "matrixsize: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    
        // [Translated]
    if (!MemorySafetyMonitor::is_operation_safe(M, N, K)) {
        std::cout << "⚠️ matrixthroughlarge，skip[Output]testwith[Output]memorysafe" << std::endl;
        return;
    }
    
    // allocatetestmatrix
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N);
    
    PerforanceMonitor monitor;
    
        // [Translated]
    std::vector<std::pair<std::string, OptimizationLevel>> methods = {
        {"fundamentalimplements", OptimizationLevel::NAIVE},
        {"loopsort", OptimizationLevel::REORDERED},
        {"chunkedoptimization", OptimizationLevel::BLOCKED},
        {"toquantization", OptimizationLevel::VECTORIZED}
    };
    
    for (const auto& method : methods) {
        // resetresultmatrix
        std::fill(C.begin(), C.end(), 0.0f);
        
        monitor.start();
        multiply(A.data(), B.data(), C.data(), M, N, K, method.second);
        long elapsed = monitor.get_elapsed_ms();
        
        double gflops = monitor.calculate_gflops(M, N, K, elapsed);
        
        std::cout << method.first << ": " << elapsed << "ms, " 
                  << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    }
    
    // testadaptiveselect
    std::fill(C.begin(), C.end(), 0.0f);
    monitor.start();
    multiply(A.data(), B.data(), C.data(), M, N, K, OptimizationLevel::ADAPTIVE);
    long elapsed = monitor.get_elapsed_ms();
    double gflops = monitor.calculate_gflops(M, N, K, elapsed);
    
    std::cout << "adaptiveselect: " << elapsed << "ms, " 
              << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    std::cout << "\navailablememory: " << MemorySafetyMonitor::get_available_memory() / (1024*1024) << " MB" << std::endl;
    std::cout << "L1cachesize: " << MemorySafetyMonitor::get_l1_cache_size() / 1024 << " KB" << std::endl;
    std::cout << "recommendedblocksize: " << AdaptiveBlockSize::calculate_safe_block_size() << std::endl;
}

} // namespace mobile_matmul
