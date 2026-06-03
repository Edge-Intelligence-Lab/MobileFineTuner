#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace ops {

enum class DType : int8_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt32 = 3,
    kInt64 = 4,
    kInt8 = 5,
    kBool = 6,
    kUInt8 = 7
};

struct DTypeUtils {
    static size_t size_of(DType dtype) {
        switch (dtype) {
            case DType::kFloat32: return sizeof(float);
            case DType::kFloat16: return sizeof(uint16_t);
            case DType::kBFloat16: return sizeof(uint16_t);
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
            case DType::kBFloat16: return "bfloat16";
            case DType::kInt32: return "int32";
            case DType::kInt64: return "int64";
            case DType::kInt8: return "int8";
            case DType::kBool: return "bool";
            case DType::kUInt8: return "uint8";
            default: return "unknown";
        }
    }

    static bool is_floating_point(DType dtype) {
        return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16;
    }

    static bool is_integer(DType dtype) {
        return dtype == DType::kInt32 || dtype == DType::kInt64 || dtype == DType::kInt8;
    }
};

inline uint16_t float32_to_fp16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent + 13);
        return static_cast<uint16_t>(sign | shifted);
    }

    if (exponent >= 31) {
        uint16_t inf_nan = (mantissa == 0) ? 0x7C00u : static_cast<uint16_t>(0x7C00u | (mantissa >> 13));
        return static_cast<uint16_t>(sign | inf_nan);
    }

    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

inline float fp16_bits_to_float32(uint16_t value) {
    uint32_t sign = (value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            exponent = exponent - 1 + 127 - 15;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float32_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even before truncating low 16 mantissa bits.
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_float32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline float lowp_to_float32(uint16_t value, DType dtype) {
    if (dtype == DType::kFloat16) {
        return fp16_bits_to_float32(value);
    }
    if (dtype == DType::kBFloat16) {
        return bf16_bits_to_float32(value);
    }
    return 0.0f;
}

constexpr DType kFloat32 = DType::kFloat32;
constexpr DType kFloat16 = DType::kFloat16;
constexpr DType kBFloat16 = DType::kBFloat16;
constexpr DType kInt32 = DType::kInt32;
constexpr DType kInt64 = DType::kInt64;
constexpr DType kInt8 = DType::kInt8;
constexpr DType kBool = DType::kBool;
constexpr DType kUInt8 = DType::kUInt8;

}
