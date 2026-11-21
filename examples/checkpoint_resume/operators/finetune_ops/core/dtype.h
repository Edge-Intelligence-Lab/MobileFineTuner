#pragma once

#include <cstdint>
#include <string>

namespace ops {

enum class DType : int8_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kInt32 = 2,
    kInt64 = 3,
    kInt8 = 4,
    kBool = 5,
    kUInt8 = 6
};

struct DTypeUtils {
    static size_t size_of(DType dtype) {
        switch (dtype) {
            case DType::kFloat32: return sizeof(float);
            case DType::kFloat16: return sizeof(uint16_t);
            case DType::kInt32: return sizeof(int32_t);
            case DType::kInt64: return sizeof(int64_t);
            case DType::kInt8: return sizeof(int8_t);
            case DType::kBool: return sizeof(bool);
            case DType::kUInt8: return sizeof(uint8_t);
            default: return 0;
        }
    }

    static std::string to_string(DType dtype) {
        switch (dtype) {
            case DType::kFloat32: return "float32";
            case DType::kFloat16: return "float16";
            case DType::kInt32: return "int32";
            case DType::kInt64: return "int64";
            case DType::kInt8: return "int8";
            case DType::kBool: return "bool";
            case DType::kUInt8: return "uint8";
            default: return "unknown";
        }
    }

    static bool is_floating_point(DType dtype) {
        return dtype == DType::kFloat32 || dtype == DType::kFloat16;
    }

    static bool is_integer(DType dtype) {
        return dtype == DType::kInt32 || dtype == DType::kInt64 || dtype == DType::kInt8;
    }
};

constexpr DType kFloat32 = DType::kFloat32;
constexpr DType kFloat16 = DType::kFloat16;
constexpr DType kInt32 = DType::kInt32;
constexpr DType kInt64 = DType::kInt64;
constexpr DType kInt8 = DType::kInt8;
constexpr DType kBool = DType::kBool;
constexpr DType kUInt8 = DType::kUInt8;

}
