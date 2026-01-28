/**
 * @file tensor.h
 * @brief Core tensor class definition for the operators framework
 * 
 * This file defines the fundamental Tensor class that serves as the primary
 * data structure for all operations in the deep learning framework.
 * It providess automatic differentiation support, memory management,
 * and a unified interface for tensor operations.
 */

#pragma once

#include "dtype.h"
#include "device.h"
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <initializer_list>
#include <cassert>
#include <stdexcept>
#include <string>

namespace ops {

// Forward declarations
class Tensor;
namespace autograd {
    class Node;
    class Engine;
    using NodePtr = std::shared_ptr<Node>;
}

/**
 * @brief Smart pointer type for Tensor objects
 * 
 * Using shared_ptr enables automatic memory management and allows
 * multiple tensors to share the same underlying data when needed.
 */
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * @brief Gradient function type for automatic differentiation
 * 
 * This function type defines how gradients are computed during backpropagation.
 * It takes the gradient of the output tensor and returns gradients for input tensors.
 */
using GradFn = std::function<std::vector<TensorPtr>(const TensorPtr&)>;

/**
 * @brief Custom exception class for tensor-related errors
 * 
 * Provides more specific error messages for tensor operations
 * compared to standard exceptions.
 */
class TensorError : public std::exception {
private:
    std::string message_;
public:
    explicit TensorError(const std::string& msg) : message_(msg) {}
    const char* what() const noexcept override { return message_.c_str(); }
};

/**
 * @brief Core tensor class for the deep learning framework
 * 
 * The Tensor class is the fundamental data structure that represents
 * multi-dimensional arrays in the framework. It supports:
 * - Automatic differentiation through gradient tracking
 * - Memory management with reference counting
 * - Various data types (float32, int32, etc.)
 * - Device placement (CPU, GPU, etc.)
 * - Shape manipulation and broadcasting
 * 
 * @note This class is designed to be similar to to PyTorch's Tensor API
 *       for familiar user experience and easy migration.
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // Default constructor
    Tensor() = default;
    /**
     * @brief Construct a tensor with specified shape, dtype, and device
     * @param shape The dimensions of the tensor
     * @param dtype The data type (default: float32)
     * @param device The device placement (default: CPU)
     */
    Tensor(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    
    /**
     * @brief Construct a tensor with existing data
     * @param shape The dimensions of the tensor
     * @param data Pointer to the data to copy
     * @param dtype The data type (default: float32)
     * @param device The device placement (default: CPU)
     */
    Tensor(const std::vector<int64_t>& shape, const void* data, DType dtype = kFloat32, Device device = kCPU);
    
    /**
     * @brief Construct a tensor that wraps existing memory (NO allocation, NO copy)
     * @param shape The dimensions of the tensor
     * @param external_data Pointer to external memory to wrap
     * @param dtype The data type
     * @param device The device placement
     * @param wrap_external_flag Must be true to activate wrapping mode
     * 
     * IMPORTANT: The caller is responsible for ensuring the external memory
     * remains valid for the lifetime of this tensor. This tensor will NOT
     * free the memory in its destructor.
     */
    Tensor(const std::vector<int64_t>& shape, void* external_data, DType dtype, Device device, bool wrap_external_flag);


    // Copy constructor and assignment operator
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // Shape and metadata accessors
    /** @brief Get the shape of the tensor */
    const std::vector<int64_t>& shape() const { return shape_; }
    /** @brief Get the size of a specific dimension */
    int64_t size(int dim) const { return shape_[static_cast<size_t>(dim)]; }
    /** @brief Get the number of dimensions */
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    /** @brief Get the total number of elements */
    int64_t numel() const;
    /** @brief Get the data type */
    DType dtype() const { return dtype_; }
    /** @brief Get the device placement */
    Device device() const { return device_; }

    void* data_ptr() { return data_; }
    const void* data_ptr() const { return data_; }

    template<typename T>
    T* data() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data() const { return static_cast<const T*>(data_); }

    template<typename T>
    T& item() {
        assert(numel() == 1);
        return *static_cast<T*>(data_);
    }

    template<typename T>
    const T& item() const {
        assert(numel() == 1);
        return *static_cast<const T*>(data_);
    }

    float item() const {
        assert(numel() == 1);
        if (dtype_ == kFloat32) return *static_cast<const float*>(data_);

        return 0.0f;
    }

    TensorPtr to(const Device& device) const;
    TensorPtr cpu() const { return to(kCPU); }
    TensorPtr cuda() const { return to(kCUDA); }

    TensorPtr reshape(const std::vector<int64_t>& new_shape) const;
    TensorPtr view(const std::vector<int64_t>& new_shape) const;
    TensorPtr squeeze(int dim = -1) const;
    TensorPtr unsqueeze(int dim) const;
    TensorPtr transpose(int dim0, int dim1) const;

    TensorPtr slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    TensorPtr operator[](int64_t index) const;

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    TensorPtr grad() const { return grad_; }
    void set_grad(const TensorPtr& grad) { grad_ = grad; }

    bool is_leaf() const { return grad_fn_ == nullptr; }

    void backward(const TensorPtr& gradient = nullptr);
    void zero_grad();

    void set_grad_fn(const GradFn& grad_fn) { grad_fn_ = grad_fn; }

    void retain_grad() { retain_grad_ = true; }
    bool retains_grad() const { return retain_grad_; }

    TensorPtr clone() const;
    TensorPtr detach() const;

    void print() const;
    std::string to_string() const;

    TensorPtr operator+(const Tensor& other) const;
    TensorPtr operator-(const Tensor& other) const;
    TensorPtr operator*(const Tensor& other) const;
    TensorPtr operator/(const Tensor& other) const;

    TensorPtr operator+(float scalar) const;
    TensorPtr operator-(float scalar) const;
    TensorPtr operator*(float scalar) const;
    TensorPtr operator/(float scalar) const;

private:
    std::vector<int64_t> shape_;
    DType dtype_ = kFloat32;
    Device device_ = kCPU;
    void* data_ = nullptr;
    size_t data_size_ = 0;
    bool owns_memory_ = true;  // 🔧 Flag whether memory is owned (for external wrapping mode)

    bool requires_grad_ = false;
    TensorPtr grad_ = nullptr;
    GradFn grad_fn_ = nullptr;  // Legacy: kept for compatibility
    bool retain_grad_ = false;
    
    // New autograd engine support
    autograd::NodePtr grad_node_ = nullptr;  // Node in computation graph

    void allocate_memory();
    void free_memory();
    void copy_data(const void* src, size_t size);
    TensorPtr shared_from_this_or_clone() const;
    
    // Allow autograd engine to access internals
    friend class autograd::Engine;
    friend class autograd::Node;
    friend class Function;
};

TensorPtr zeros(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr ones(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr randn(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr rand(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr empty(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr full(const std::vector<int64_t>& shape, float value, DType dtype = kFloat32, Device device = kCPU);

TensorPtr from_blob(void* data, const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
TensorPtr tensor(const std::vector<float>& data, DType dtype = kFloat32, Device device = kCPU);
TensorPtr tensor(std::initializer_list<float> data, DType dtype = kFloat32, Device device = kCPU);

TensorPtr cast(const TensorPtr& tensor, DType dtype);

TensorPtr operator+(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator-(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator*(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator/(const TensorPtr& a, const TensorPtr& b);

TensorPtr operator+(const TensorPtr& tensor, float scalar);
TensorPtr operator-(const TensorPtr& tensor, float scalar);
TensorPtr operator*(const TensorPtr& tensor, float scalar);
TensorPtr operator/(const TensorPtr& tensor, float scalar);

TensorPtr operator+(float scalar, const TensorPtr& tensor);
TensorPtr operator-(float scalar, const TensorPtr& tensor);
TensorPtr operator*(float scalar, const TensorPtr& tensor);
TensorPtr operator/(float scalar, const TensorPtr& tensor);

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

}
