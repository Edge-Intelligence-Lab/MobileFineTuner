/**
 * @file tensor.cpp
 * @brief Implementation of the core Tensor class
 * 
 * This file contains the implementsation of all Tensor class methods,
 * including constructors, operators, memory management, and utility functions.
 * It also providess tensor creation functions and automatic differentiation support.
 */

#include "tensor.h"
#include "memory_manager.h"
#include "../memory/arena_allocator.h"
#include "ops.h"
#include "backward_functions.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>

namespace ops {

namespace {

    /**
     * @brief Compute the total number of elements in a tensor
     * @param shape The shape vector
     * @return Total number of elements
     */
    int64_t compute_numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    }

    /**
     * @brief Get the size in bytes for a given data type
     * @param dtype The data type
     * @return Size in bytes
     */
    size_t dtype_size(DType dtype) {
        switch (dtype) {
            case kFloat32: return sizeof(float);
            case kFloat16: return sizeof(uint16_t);
            case kInt32: return sizeof(int32_t);
            case kInt64: return sizeof(int64_t);
            case kInt8: return sizeof(int8_t);
            case kBool: return sizeof(bool);
            default: return sizeof(float);
        }
    }

    void validate_shape(const std::vector<int64_t>& shape) {
        for (auto dim : shape) {
            if (dim < 0) {
                throw TensorError("Shape dimensions must be non-negative");
            }
        }
    }

    // Unused currently, but kept for potential future use
    [[maybe_unused]]
    int64_t compute_linear_index(const std::vector<int64_t>& indices, const std::vector<int64_t>& shape) {
        if (indices.size() != shape.size()) {
            throw TensorError("Index dimension mismatch");
        }

        int64_t linear_index = 0;
        int64_t stride = 1;

        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw TensorError("Index out of bounds");
            }
            linear_index += indices[i] * stride;
            stride *= shape[i];
        }

        return linear_index;
    }
}

TensorPtr Tensor::shared_from_this_or_clone() const {
    try {
        return std::const_pointer_cast<Tensor>(shared_from_this());
    } catch (const std::bad_weak_ptr&) {
        return std::make_shared<Tensor>(*this);
    }
}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    validate_shape(shape);
    allocate_memory();
}

Tensor::Tensor(const std::vector<int64_t>& shape, const void* data, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    validate_shape(shape);
    allocate_memory();
    if (data) {
        copy_data(data, data_size_);
    }
}

Tensor::Tensor(const std::vector<int64_t>& shape, void* external_data, DType dtype, Device device, bool wrap_external_flag)
    : shape_(shape), dtype_(dtype), device_(device), owns_memory_(false) {
    (void)wrap_external_flag;  // Parameter used only to disambiguate overloads
    validate_shape(shape);
    
    // Use external memory directly without allocation
    int64_t n = compute_numel(shape);
    data_size_ = n * dtype_size(dtype);
    data_ = external_data;
    
    // Do not call allocate_memory()
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_),
      requires_grad_(other.requires_grad_), retain_grad_(other.retain_grad_) {
    allocate_memory();
    if (other.data_) {
        copy_data(other.data_, data_size_);
    }

}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        free_memory();

        shape_ = other.shape_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        retain_grad_ = other.retain_grad_;

        allocate_memory();
        if (other.data_) {
            copy_data(other.data_, data_size_);
        }

        grad_ = nullptr;
        grad_fn_ = nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_), data_size_(other.data_size_),
      requires_grad_(other.requires_grad_), grad_(std::move(other.grad_)),
      grad_fn_(std::move(other.grad_fn_)), retain_grad_(other.retain_grad_) {
    other.data_ = nullptr;
    other.data_size_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_memory();

        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        data_size_ = other.data_size_;
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        grad_fn_ = std::move(other.grad_fn_);
        retain_grad_ = other.retain_grad_;

        other.data_ = nullptr;
        other.data_size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    free_memory();
}

int64_t Tensor::numel() const {
    return compute_numel(shape_);
}

void Tensor::allocate_memory() {
    if (numel() == 0) {
        data_ = nullptr;
        data_size_ = 0;
        return;
    }

    data_size_ = numel() * dtype_size(dtype_);

    // Memory allocation strategy:
    // 1. Arena mode: partitioned memory manager (recommended)
    // 2. Default mode: direct malloc (simple but may fragment)
    
    #ifdef USE_ARENA_ALLOCATOR
    data_ = memory::ArenaManager::instance().allocate(data_size_);
    #else
    #ifdef DISABLE_MEMORY_POOL
    // Use system allocation directly (for RSS diagnostics)
    data_ = std::malloc(data_size_);
    if (data_) {
        std::memset(data_, 0, data_size_);
    }
    #else
    // Use global MemoryManager pool for unified cleanup and return to OS
    data_ = MemoryManager::instance().allocate(data_size_);
    if (data_) {
        std::memset(data_, 0, data_size_);
    }
    #endif
    #endif
    
    if (!data_) {
        throw TensorError("Failed to allocate memory");
    }
}

void Tensor::free_memory() {
    if (data_ && owns_memory_) {
        // Release only memory we own
        #ifdef USE_ARENA_ALLOCATOR
        memory::ArenaManager::instance().free(data_, data_size_);
        #else
        #ifdef DISABLE_MEMORY_POOL
        std::free(data_);
        #else
        // Return memory to MemoryManager (can be reclaimed to OS as needed)
        MemoryManager::instance().deallocate(data_, data_size_);
        #endif
        #endif
        
        data_ = nullptr;
        data_size_ = 0;
    } else if (data_ && !owns_memory_) {
        // External memory: only clear pointer
        data_ = nullptr;
        data_size_ = 0;
    }
}

void Tensor::copy_data(const void* src, size_t size) {
    if (!data_ || !src || size == 0) return;

    size_t copy_size = std::min(size, data_size_);
    std::memcpy(data_, src, copy_size);
}

TensorPtr Tensor::to(const Device& device) const {
    if (device == device_) {
        return clone();
    }

    if (!device_.is_cpu() || !device.is_cpu()) {
        throw TensorError("Tensor::to: only CPU<->CPU copies are implemented");
    }

    auto result = std::make_shared<Tensor>(shape_, dtype_, device);
    result->set_requires_grad(requires_grad_);
    if (data_ && result->data_ptr()) {
        std::memcpy(result->data_ptr(), data_, data_size_);
    }
    return result;
}

TensorPtr Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = compute_numel(new_shape);
    if (new_numel != numel()) {
        throw TensorError("Cannot reshape tensor: element count mismatch");
    }

    auto result = std::make_shared<Tensor>(new_shape, data_, dtype_, device_);

    if (requires_grad_) {
        result->set_requires_grad(true);
        
        // [Translated comment removed - see documentation]
        auto original_shape = shape_;
        result->set_grad_fn([original_shape](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = grad_output->reshape(original_shape);
            return {grad_input};
        });
    }

    return result;
}


TensorPtr Tensor::view(const std::vector<int64_t>& new_shape) const {

    return reshape(new_shape);
}

TensorPtr Tensor::transpose(int dim0, int dim1) const {
    if (ndim() < 2) {
        throw TensorError("transpose requires at least 2 dimensions");
    }

    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();

    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw TensorError("transpose dimension out of range");
    }

    auto new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    auto result = std::make_shared<Tensor>(new_shape, dtype_, device_);

    const float* src_data = data<float>();
    float* dst_data = result->data<float>();

    if (ndim() == 2 && dim0 == 0 && dim1 == 1) {

        int64_t rows = shape_[0];
        int64_t cols = shape_[1];

        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                dst_data[j * rows + i] = src_data[i * cols + j];
            }
        }
    } else {

        std::vector<int64_t> src_indices(ndim(), 0);

        for (int64_t linear_idx = 0; linear_idx < numel(); ++linear_idx) {

            int64_t temp = linear_idx;
            for (int d = ndim() - 1; d >= 0; --d) {
                src_indices[d] = temp % shape_[d];
                temp /= shape_[d];
            }

            std::swap(src_indices[dim0], src_indices[dim1]);

            int64_t dst_linear_idx = 0;
            int64_t stride = 1;
            for (int d = ndim() - 1; d >= 0; --d) {
                dst_linear_idx += src_indices[d] * stride;
                stride *= new_shape[d];
            }

            dst_data[dst_linear_idx] = src_data[linear_idx];
        }
    }

    if (requires_grad_) {
        result->set_requires_grad(true);
        
                // [Translated]
        result->set_grad_fn([dim0, dim1](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            // [Translated comment removed - see documentation]
            auto grad_input = grad_output->transpose(dim0, dim1);
            return {grad_input};
        });
    }

    return result;
}

TensorPtr Tensor::squeeze(int dim) const {
    auto new_shape = shape_;

    if (dim == -1) {

        new_shape.erase(
            std::remove(new_shape.begin(), new_shape.end(), 1),
            new_shape.end()
        );
    } else {

        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim()) {
            throw TensorError("squeeze dimension out of range");
        }
        if (shape_[dim] == 1) {
            new_shape.erase(new_shape.begin() + dim);
        }
    }

    return reshape(new_shape);
}

TensorPtr Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    if (dim < 0 || dim > ndim()) {
        throw TensorError("unsqueeze dimension out of range");
    }

    auto new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);

    return reshape(new_shape);
}

TensorPtr Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw TensorError("slice dimension out of range");
    }
    
    int64_t dim_size = shape_[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(int64_t(0), std::min(end, dim_size));
    
    if (step != 1) {
        throw TensorError("slice with step != 1 not implemented");
    }
    
    int64_t slice_len = end - start;
    if (slice_len <= 0) {
        throw TensorError("slice length must be positive");
    }
    
    // Compute new shape
    auto new_shape = shape_;
    new_shape[dim] = slice_len;
    
    // Create result tensor
    auto result = std::make_shared<Tensor>(new_shape, dtype_, device_);
    
    // Copy data (simplified: contiguous slices only)
    const float* src_data = data<float>();
    float* dst_data = result->data<float>();
    
    // Compute strides
    std::vector<int64_t> strides(ndim());
    strides[ndim() - 1] = 1;
    for (int i = ndim() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
    }
    
    // 复制切片数据
    int64_t dst_idx = 0;
    std::function<void(int, int64_t)> copy_slice;
    copy_slice = [&](int d, int64_t src_offset) {
        if (d == ndim()) {
            dst_data[dst_idx++] = src_data[src_offset];
            return;
        }
        
        int64_t dim_start = (d == dim) ? start : 0;
        int64_t dim_end = (d == dim) ? end : shape_[d];
        
        for (int64_t i = dim_start; i < dim_end; ++i) {
            copy_slice(d + 1, src_offset + i * strides[d]);
        }
    };
    
    copy_slice(0, 0);
    
    if (requires_grad_) {
        result->set_requires_grad(true);
        auto input_ptr = shared_from_this_or_clone();
        #ifdef USE_NEW_AUTOGRAD_ENGINE
        auto backward_fn = std::make_shared<SliceBackward>(input_ptr, dim, start, slice_len);
        autograd::Engine::instance().register_node(result, {input_ptr}, backward_fn);
        #else
        result->set_grad_fn([input_ptr, dim, start, slice_len](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            auto grad_input = zeros(input_ptr->shape(), input_ptr->dtype(), input_ptr->device());
            const float* grad_out = grad_output->data<float>();
            float* grad_in = grad_input->data<float>();

            const auto& input_shape = input_ptr->shape();
            std::vector<int64_t> in_strides(input_shape.size());
            in_strides.back() = 1;
            for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; --i) {
                in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
            }

            auto sliced_shape = grad_output->shape();
            std::vector<int64_t> out_strides(sliced_shape.size());
            out_strides.back() = 1;
            for (int i = static_cast<int>(sliced_shape.size()) - 2; i >= 0; --i) {
                out_strides[i] = out_strides[i + 1] * sliced_shape[i + 1];
            }

            std::function<void(int,int64_t,int64_t)> scatter = [&](int d, int64_t out_offset, int64_t in_offset) {
                if (d == static_cast<int>(input_shape.size())) {
                    grad_in[in_offset] += grad_out[out_offset];
                    return;
                }
                int64_t steps = (d == dim) ? slice_len : input_shape[d];
                int64_t in_start = (d == dim) ? start : 0;
                for (int64_t i = 0; i < steps; ++i) {
                    scatter(d + 1,
                            out_offset + i * out_strides[d],
                            in_offset + (in_start + i) * in_strides[d]);
                }
            };
            scatter(0, 0, 0);
            return {grad_input};
        });
        #endif
    }
    
    return result;
}

TensorPtr Tensor::operator[](int64_t index) const {
    if (shape_.empty()) {
        throw TensorError("operator[] cannot index scalar tensor");
    }
    int64_t dim0 = shape_[0];
    if (index < 0) index += dim0;
    if (index < 0 || index >= dim0) {
        throw TensorError("operator[] index out of range");
    }
    auto sliced = slice(0, index, index + 1);
    return sliced->squeeze(0);
}

TensorPtr Tensor::clone() const {
    auto result = std::make_shared<Tensor>(shape_, data_, dtype_, device_);
    result->set_requires_grad(requires_grad_);
    return result;
}

TensorPtr Tensor::detach() const {
    auto result = std::make_shared<Tensor>(shape_, data_, dtype_, device_);

    return result;
}

void Tensor::backward(const TensorPtr& gradient) {
    if (!requires_grad_) {
        return;
    }
    
    #ifdef USE_NEW_AUTOGRAD_ENGINE
    // Forward to the new engine
    #ifdef AUTOGRAD_DEBUG
    // Debug only
    #endif
    
    // Need to find the shared_ptr for this tensor - use grad_node_ if available
    TensorPtr this_ptr;
    if (grad_node_ && grad_node_->tensor) {
        this_ptr = grad_node_->tensor;
    } else {
        // Fallback: create a temporary shared_ptr (not ideal but works for testing)
        // In production, tensors should always be created as shared_ptr
        this_ptr = std::shared_ptr<Tensor>(this, [](Tensor*){/* no-op deleter */});
    }
    
    std::vector<TensorPtr> outputs = {this_ptr};
    std::vector<TensorPtr> grads = gradient ? std::vector<TensorPtr>{gradient} : std::vector<TensorPtr>{};
    autograd::Engine::instance().run_backward(outputs, grads);
    return;
    #endif
    
    // Legacy recursive backward (kept for compatibility)
    TensorPtr grad_tensor = gradient;
    if (!grad_tensor) {
        if (numel() != 1) {
            throw TensorError("grad can be implicitly created only for scalar outputs");
        }
        grad_tensor = ones(shape_, dtype_, device_);
    }

    if (grad_tensor->shape() != shape_) {
        throw TensorError("gradient shape does not match tensor shape");
    }

    if (!grad_) {
        grad_ = grad_tensor->clone();
    } else {
        auto accumulated_grad = zeros(grad_->shape(), grad_->dtype(), grad_->device());

        const float* old_grad_data = grad_->data<float>();
        const float* new_grad_data = grad_tensor->data<float>();
        float* accumulated_data = accumulated_grad->data<float>();

        for (int64_t i = 0; i < grad_->numel(); ++i) {
            accumulated_data[i] = old_grad_data[i] + new_grad_data[i];
        }

        grad_ = accumulated_grad;
    }

    if (grad_fn_) {
        grad_fn_(grad_tensor);
    }
}

void Tensor::zero_grad() {
    if (grad_) {

        std::memset(grad_->data_ptr(), 0, grad_->data_size_);
    }
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << static_cast<int>(dtype_);
    oss << ", device=" << static_cast<int>(device_.type());
    if (requires_grad_) oss << ", requires_grad=True";
    oss << ")";
    return oss.str();
}

TensorPtr Tensor::operator+(const Tensor& other) const {
    return add(shared_from_this_or_clone(), other.shared_from_this_or_clone());
}

TensorPtr Tensor::operator-(const Tensor& other) const {
    return sub(shared_from_this_or_clone(), other.shared_from_this_or_clone());
}

TensorPtr Tensor::operator*(const Tensor& other) const {
    return mul(shared_from_this_or_clone(), other.shared_from_this_or_clone());
}

TensorPtr Tensor::operator/(const Tensor& other) const {
    return div(shared_from_this_or_clone(), other.shared_from_this_or_clone());
}

TensorPtr Tensor::operator+(float scalar) const {
    return add(shared_from_this_or_clone(), scalar);
}

TensorPtr Tensor::operator-(float scalar) const {
    return sub(shared_from_this_or_clone(), scalar);
}

TensorPtr Tensor::operator*(float scalar) const {
    return mul(shared_from_this_or_clone(), scalar);
}

TensorPtr Tensor::operator/(float scalar) const {
    return div(shared_from_this_or_clone(), scalar);
}

TensorPtr zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    // 🔧 FIX: 确保所有类型都清零
    if (dtype == kFloat32 || dtype == kFloat16) {
        std::memset(tensor->data_ptr(), 0, tensor->numel() * dtype_size(dtype));
    } else if (dtype == kInt32) {
        std::memset(tensor->data_ptr(), 0, tensor->numel() * sizeof(int32_t));
    } else if (dtype == kInt64) {
        std::memset(tensor->data_ptr(), 0, tensor->numel() * sizeof(int64_t));
    } else {
        std::memset(tensor->data_ptr(), 0, tensor->numel() * dtype_size(dtype));
    }

    return tensor;
}

TensorPtr ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        float* data = tensor->data<float>();
        std::fill_n(data, tensor->numel(), 1.0f);
    } else if (dtype == kInt32) {
        int32_t* data = tensor->data<int32_t>();
        std::fill_n(data, tensor->numel(), 1);
    }

    return tensor;
}

TensorPtr randn(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        float* data = tensor->data<float>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            data[i] = dist(gen);
        }
    }

    return tensor;
}

TensorPtr rand(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        float* data = tensor->data<float>();
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            data[i] = dist(gen);
        }
    }

    return tensor;
}

TensorPtr uniform(const std::vector<int64_t>& shape, float low, float high,
                  DType dtype, Device device) {
    if (low > high) {
        throw TensorError("uniform: low must be <= high");
    }
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);
    if (dtype != kFloat32) {
        throw TensorError("uniform: only float32 tensors are supported");
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = dist(gen);
    }
    return tensor;
}

TensorPtr empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    return std::make_shared<Tensor>(shape, dtype, device);
}

TensorPtr full(const std::vector<int64_t>& shape, float value, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);

    if (dtype == kFloat32) {
        float* data = tensor->data<float>();
        std::fill_n(data, tensor->numel(), value);
    }

    return tensor;
}

TensorPtr from_blob(void* data, const std::vector<int64_t>& shape, DType dtype, Device device) {
    if (!data) {
        throw TensorError("from_blob: data pointer must not be null");
    }
    return std::make_shared<Tensor>(shape, data, dtype, device, true);
}

TensorPtr tensor(const std::vector<float>& data, DType dtype, Device device) {
    std::vector<int64_t> shape = {static_cast<int64_t>(data.size())};
    return std::make_shared<Tensor>(shape, data.data(), dtype, device);
}

TensorPtr tensor(std::initializer_list<float> data, DType dtype, Device device) {
    std::vector<float> vec_data(data);
    return tensor(vec_data, dtype, device);
}

TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) {
    return add(a, b);
}

TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) {
    return sub(a, b);
}

TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) {
    return mul(a, b);
}

TensorPtr operator/(const TensorPtr& a, const TensorPtr& b) {
    return div(a, b);
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}


}
