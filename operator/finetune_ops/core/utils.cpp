/**
 * @file utils.cpp
 * @brief Implementation of utility functions for the operators framework
 * 
 * This file contains the implementsation of utility functions including
 * random number generation, memory management, tensor operations,
 * and debugging utilities.
 */

#include "utils.h"
#include <fstream>
#include <thread>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <sys/resource.h>
#include <iostream>
#include <future>
#include <queue>

namespace ops {
namespace utils {

// Thread-local random number generator instances
thread_local std::mt19937 Random::generator_;
thread_local uint64_t Random::seed_ = std::random_device()();

void Random::manual_seed(uint64_t seed) {
    seed_ = seed;
    generator_.seed(seed);
}

uint64_t Random::get_seed() {
    return seed_;
}

std::mt19937& Random::generator() {
    return generator_;
}

std::vector<int> broadcast_shape(const std::vector<int>& shape1,
                               const std::vector<int>& shape2) {
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<int> result(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t idx1 = shape1.size() > i ? shape1.size() - 1 - i : 0;
        size_t idx2 = shape2.size() > i ? shape2.size() - 1 - i : 0;

        int dim1 = shape1.size() > i ? shape1[idx1] : 1;
        int dim2 = shape2.size() > i ? shape2[idx2] : 1;

        if (dim1 == dim2) {
            result[max_dims - 1 - i] = dim1;
        } else if (dim1 == 1) {
            result[max_dims - 1 - i] = dim2;
        } else if (dim2 == 1) {
            result[max_dims - 1 - i] = dim1;
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }
    }

    return result;
}

bool can_broadcast(const std::vector<int>& shape1,
                  const std::vector<int>& shape2) {
    try {
        broadcast_shape(shape1, shape2);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

std::vector<int> expand_dims(const std::vector<int>& shape, int axis) {
    if (axis < 0) {
        axis += shape.size() + 1;
    }
    if (axis < 0 || axis > static_cast<int>(shape.size())) {
        throw std::runtime_error("Invalid axis for expand_dims");
    }

    std::vector<int> result;
    result.reserve(shape.size() + 1);

    for (int i = 0; i < axis; ++i) {
        result.push_back(shape[i]);
    }
    result.push_back(1);
    for (size_t i = axis; i < shape.size(); ++i) {
        result.push_back(shape[i]);
    }

    return result;
}

size_t get_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    return 0;
}

void save_tensor_binary(const Tensor& tensor, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    const auto& shape = tensor.shape();
    int32_t num_dims = static_cast<int32_t>(shape.size());
    file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
    file.write(reinterpret_cast<const char*>(shape.data()),
               num_dims * sizeof(int32_t));

    int32_t dtype = static_cast<int32_t>(tensor.dtype());
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

    const float* data = tensor.data<float>();
    file.write(reinterpret_cast<const char*>(data),
               tensor.numel() * sizeof(float));
}

TensorPtr load_tensor_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    int32_t num_dims;
    file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
    std::vector<int> shape(num_dims);
    file.read(reinterpret_cast<char*>(shape.data()),
              num_dims * sizeof(int32_t));

    int32_t dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

    size_t data_size = std::accumulate(shape.begin(), shape.end(), 1LL,
                                     std::multiplies<size_t>());
    std::vector<float> data(data_size);
    file.read(reinterpret_cast<char*>(data.data()),
              data_size * sizeof(float));

    std::vector<int64_t> shape64(shape.begin(), shape.end());
    return std::make_shared<Tensor>(shape64, data.data(),
                                  static_cast<DType>(dtype));
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string tensor_to_string(const Tensor& tensor,
                           const std::string& indent) {
    std::ostringstream oss;
    oss << indent << "Tensor(shape=" << shape_to_string(tensor.shape());

    oss << ", dtype=";
    switch (tensor.dtype()) {
        case kFloat32: oss << "float32"; break;
        case kInt32: oss << "int32"; break;
        case kInt8: oss << "int8"; break;
        case kFloat16: oss << "float16"; break;
        case kInt64: oss << "int64"; break;
        case kBool: oss << "bool"; break;
        case kUInt8: oss << "uint8"; break;
    }

    const float* data = tensor.data<float>();
    if (tensor.numel() == 0) {
        oss << ", data=[])";
        return oss.str();
    }

    oss << ", data=\n" << indent << "  [";
    size_t max_elements = 10;
    for (size_t i = 0; i < std::min(static_cast<size_t>(tensor.numel()), max_elements); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << data[i];
    }
    if (static_cast<size_t>(tensor.numel()) > max_elements) {
        oss << ", ...";
    }
    oss << "])";

    return oss.str();
}

void print_memory_stats() {
    size_t memory_usage = get_memory_usage();
    std::cout << "Memory usage: "
              << std::fixed << std::setprecision(2)
              << (memory_usage / 1024.0 / 1024.0) << " MB\n";
}

Timer::Timer() : running_(false) {}

Timer::~Timer() {
    if (running_) {
        stop();
    }
}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
}

double Timer::elapsed_ms() const {
    auto end = running_ ?
        std::chrono::high_resolution_clock::now() : end_time_;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start_time_);
    return duration.count() / 1000.0;
}

void Timer::reset() {
    running_ = false;
}

Error::Error(const std::string& message)
    : std::runtime_error(message) {}

Error::Error(const char* message)
    : std::runtime_error(message) {}

ShapeError::ShapeError(const std::string& message)
    : Error(message) {}

ShapeError::ShapeError(const char* message)
    : Error(message) {}

namespace {
bool debug_mode_ = false;
}

void set_debug_mode(bool enabled) {
    debug_mode_ = enabled;
}

bool is_debug_mode() {
    return debug_mode_;
}

void debug_print(const std::string& message) {
    if (debug_mode_) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
}

void debug_tensor(const std::string& name, const Tensor& tensor) {
    if (debug_mode_) {
        std::cout << "[DEBUG] " << name << ":\n"
                  << tensor_to_string(tensor, "  ") << std::endl;
    }
}

namespace {
bool simd_enabled_ = true;
}

bool is_simd_enabled() {
    return simd_enabled_;
}

void set_simd_enabled(bool enabled) {
    simd_enabled_ = enabled;
}

int get_simd_alignment() {
    return 32;
}

class ThreadPool::Impl {
public:
    explicit Impl(size_t num_threads)
        : stop_(false) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~Impl() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f),
                     std::forward<Args>(args)...)
        );

        auto future = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();

        return future;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] {
            return tasks_.empty();
        });
    }

    size_t get_num_threads() const {
        return workers_.size();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

ThreadPool::ThreadPool(size_t num_threads)
    : impl_(std::make_unique<Impl>(num_threads)) {}

ThreadPool::~ThreadPool() = default;

template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) {
    return impl_->submit(std::forward<F>(f),
                        std::forward<Args>(args)...);
}

void ThreadPool::wait_all() {
    impl_->wait_all();
}

size_t ThreadPool::get_num_threads() const {
    return impl_->get_num_threads();
}

}
}
