/**
 * @file utils.h
 * @brief Utility functions and helper classes for the operators framework
 * 
 * This file providess utility functions for tensor operations, random number
 * generation, memory management, and other common tasks used throughout
 * the framework.
 */

#pragma once

#include "tensor.h"
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <chrono>

namespace ops {
namespace utils {

/**
 * @brief Thread-safe random number generator class
 * 
 * Provides a thread-local random number generator that can be seeded
 * for reproducible results. Each thread maintains its own generator
 * to avoid race conditions.
 */
class Random {
public:
    /** @brief Set the random seed for reproducible results */
    static void manual_seed(uint64_t seed);
    /** @brief Get the current random seed */
    static uint64_t get_seed();
    /** @brief Get the random number generator instance */
    static std::mt19937& generator();

private:
    static thread_local std::mt19937 generator_;
    static thread_local uint64_t seed_;
};

std::vector<int> broadcast_shape(const std::vector<int>& shape1,
                               const std::vector<int>& shape2);
bool can_broadcast(const std::vector<int>& shape1,
                  const std::vector<int>& shape2);
std::vector<int> expand_dims(const std::vector<int>& shape,
                           int axis);

size_t get_memory_usage();

template<typename T>
std::vector<T> cast_vector(const std::vector<float>& data);

template<typename T>
std::vector<float> cast_to_float(const std::vector<T>& data);

void save_tensor_binary(const Tensor& tensor,
                       const std::string& path);
TensorPtr load_tensor_binary(const std::string& path);

std::string shape_to_string(const std::vector<int64_t>& shape);
std::string tensor_to_string(const Tensor& tensor,
                           const std::string& indent = "");
void print_memory_stats();

class Timer {
public:
    Timer();
    ~Timer();

    void start();
    void stop();
    double elapsed_ms() const;
    void reset();

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message);
    explicit Error(const char* message);
};

class ShapeError : public Error {
public:
    explicit ShapeError(const std::string& message);
    explicit ShapeError(const char* message);
};

void set_debug_mode(bool enabled);
bool is_debug_mode();
void debug_print(const std::string& message);
void debug_tensor(const std::string& name,
                 const Tensor& tensor);

bool is_simd_enabled();
void set_simd_enabled(bool enabled);
int get_simd_alignment();

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args);

    void wait_all();
    size_t get_num_threads() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}
}