/**
 * @file fp16_utils.h
 * @brief FP16 conversion utilities and helper functions
 * 
 * Standard IEEE 754 FP16 (binary16) conversion following PyTorch conventions
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace ops {
namespace fp16 {

/**
 * @brief Convert FP32 to FP16 (IEEE 754 binary16)
 * Standard rounding: round-to-nearest-even
 */
inline uint16_t float32_to_float16(float value) {
#ifdef __ARM_NEON
    // Use ARM NEON hardware FP16 conversion
    float16_t f16 = vcvt_f16_f32(vdupq_n_f32(value))[0];
    return *reinterpret_cast<uint16_t*>(&f16);
#else
    // Software fallback: IEEE 754 compliant conversion
    uint32_t f32_bits = *reinterpret_cast<uint32_t*>(&value);
    
    uint32_t sign = (f32_bits >> 16) & 0x8000;
    int32_t exponent = ((f32_bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32_bits & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == 128) {  // Inf or NaN
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00);  // Inf
        } else {
            return static_cast<uint16_t>(sign | 0x7E00);  // NaN
        }
    }
    
    if (exponent < -14) {  // Denormal or underflow
        if (exponent < -25) return static_cast<uint16_t>(sign);  // Underflow to zero
        
        // Denormal number
        mantissa |= 0x800000;  // Add implicit 1
        int shift = -exponent - 14;
        mantissa >>= shift;
        
        // Round to nearest even
        if ((mantissa & 0x1000) && ((mantissa & 0x2001) || shift < 13)) {
            mantissa += 0x1000;
        }
        
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    
    if (exponent > 15) {  // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    
    // Normal number
    uint32_t f16_exp = (exponent + 15) << 10;
    uint32_t f16_mantissa = mantissa >> 13;
    
    // Round to nearest even
    if ((mantissa & 0x1000) && ((mantissa & 0x2001))) {
        f16_mantissa += 1;
        if (f16_mantissa == 0x400) {  // Overflow mantissa
            f16_exp += 0x400;
            f16_mantissa = 0;
        }
    }
    
    return static_cast<uint16_t>(sign | f16_exp | f16_mantissa);
#endif
}

/**
 * @brief Convert FP16 to FP32 (lossless)
 */
inline float float16_to_float32(uint16_t f16_bits) {
#ifdef __ARM_NEON
    // Use ARM NEON hardware FP16 conversion
    float16_t f16 = *reinterpret_cast<float16_t*>(&f16_bits);
    return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(f16)), 0);
#else
    // Software fallback: IEEE 754 compliant conversion
    uint32_t sign = (f16_bits & 0x8000) << 16;
    int32_t exponent = (f16_bits >> 10) & 0x1F;
    uint32_t mantissa = f16_bits & 0x3FF;
    
    // Handle special cases
    if (exponent == 0) {  // Zero or denormal
        if (mantissa == 0) {
            uint32_t result_bits = sign;
            return *reinterpret_cast<float*>(&result_bits);
        }
        
        // Denormal: normalize it
        exponent = 1;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
    } else if (exponent == 31) {  // Inf or NaN
        uint32_t result_bits = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result_bits);
    }
    
    // Normal number
    uint32_t f32_exp = (exponent + 112) << 23;
    uint32_t f32_mantissa = mantissa << 13;
    uint32_t result_bits = sign | f32_exp | f32_mantissa;
    
    return *reinterpret_cast<float*>(&result_bits);
#endif
}

/**
 * @brief Batch convert FP32 array to FP16 array
 */
inline void convert_fp32_to_fp16(const float* src, uint16_t* dst, size_t count) {
#ifdef __ARM_NEON
    // Vectorized conversion using NEON
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t f32_vec = vld1q_f32(src + i);
        float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
        vst1_u16(dst + i, vreinterpret_u16_f16(f16_vec));
    }
    // Handle remaining elements
    for (; i < count; ++i) {
        dst[i] = float32_to_float16(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float32_to_float16(src[i]);
    }
#endif
}

/**
 * @brief Batch convert FP16 array to FP32 array
 */
inline void convert_fp16_to_fp32(const uint16_t* src, float* dst, size_t count) {
#ifdef __ARM_NEON
    // Vectorized conversion using NEON
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float16x4_t f16_vec = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t f32_vec = vcvt_f32_f16(f16_vec);
        vst1q_f32(dst + i, f32_vec);
    }
    // Handle remaining elements
    for (; i < count; ++i) {
        dst[i] = float16_to_float32(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float16_to_float32(src[i]);
    }
#endif
}

} // namespace fp16
} // namespace ops
