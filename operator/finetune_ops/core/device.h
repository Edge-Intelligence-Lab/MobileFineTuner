#pragma once

#include <string>
#include <memory>

namespace ops {

enum class DeviceType : int8_t {
    kCPU = 0,
    kCUDA = 1,
    kMetal = 2
};

class Device {
public:
    Device(DeviceType type, int index = 0) : type_(type), index_(index) {}

    DeviceType type() const { return type_; }
    int index() const { return index_; }

    bool is_cpu() const { return type_ == DeviceType::kCPU; }
    bool is_cuda() const { return type_ == DeviceType::kCUDA; }
    bool is_metal() const { return type_ == DeviceType::kMetal; }

    std::string to_string() const {
        switch (type_) {
            case DeviceType::kCPU: return "cpu";
            case DeviceType::kCUDA: return "cuda:" + std::to_string(index_);
            case DeviceType::kMetal: return "metal:" + std::to_string(index_);
            default: return "unknown";
        }
    }

    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    DeviceType type_;
    int index_;
};

class DeviceManager {
public:
    static Device best_available() {
        #ifdef CUDA_AVAILABLE
        if (cuda_available()) {
            return Device(DeviceType::kCUDA, 0);
        }
        #endif

        #ifdef METAL_AVAILABLE
        if (metal_available()) {
            return Device(DeviceType::kMetal, 0);
        }
        #endif

        return Device(DeviceType::kCPU, 0);
    }

    static Device cpu() { return Device(DeviceType::kCPU, 0); }
    static Device cuda(int index = 0) { return Device(DeviceType::kCUDA, index); }
    static Device metal(int index = 0) { return Device(DeviceType::kMetal, index); }

    static bool cuda_available() {
        #ifdef CUDA_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }

    static bool metal_available() {
        #ifdef METAL_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }

    static int cuda_device_count() {
        #ifdef CUDA_AVAILABLE
        return 1;
        #else
        return 0;
        #endif
    }
};

const Device kCPU = Device(DeviceType::kCPU, 0);
const Device kCUDA = Device(DeviceType::kCUDA, 0);
const Device kMetal = Device(DeviceType::kMetal, 0);

}
