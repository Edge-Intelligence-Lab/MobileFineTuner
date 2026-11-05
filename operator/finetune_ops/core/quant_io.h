/**
 * @file quant_io.h
 * @brief Serialization for Q8_0-quantized GPT-2 models.
 *
 * The checkpoint is a single self-contained file: quantized linear weights are
 * stored as Q8_0 blocks (34 bytes per 32 values), everything else (embeddings,
 * LayerNorm, biases) stays fp32. Loading fills the model directly, so the fp32
 * base weights are never materialised.
 *
 * Layout (little endian):
 *   magic   "MFTQ8\0\0\0"   8 bytes
 *   version u32             currently 1
 *   count   u32             number of tensors
 *   per tensor:
 *     name_len u32, name bytes
 *     is_q8    u8
 *     if is_q8:  k u32, m u32, nblocks u64, nblocks * 34 bytes
 *     else:      ndim u32, shape[ndim] i64, count u64, count * 4 bytes (fp32)
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "quant_ops.h"

namespace ops {

struct QuantTensor {
    std::string name;

    // Q8_0 tensor: base weight [K, M] stored as blocks.
    bool is_q8 = false;
    int64_t k = 0;
    int64_t m = 0;
    std::vector<block_q8_0> q8;

    // Plain fp32 tensor.
    std::vector<int64_t> shape;
    std::vector<float> f32;
};

// Write all tensors to `path`. Returns false and fills `err` on failure.
bool write_quant_model(const std::string& path, const std::vector<QuantTensor>& tensors,
                       std::string* err = nullptr);

// Read a file produced by write_quant_model(). Returns false on bad magic,
// unsupported version, or a truncated/corrupt payload.
bool read_quant_model(const std::string& path, std::vector<QuantTensor>& out,
                      std::string* err = nullptr);

}  // namespace ops
