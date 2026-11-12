#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace ops {
namespace detail {

inline std::vector<int64_t> dense_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t d = shape.size(); d-- > 0;) {
        strides[d] = stride;
        stride *= shape[d];
    }
    return strides;
}

// Source strides are expressed in destination axis order. Both buffers are
// non-overlapping; singleton axes do not interrupt a contiguous trailing block.
inline void copy_permuted(const void* source, void* destination,
                          const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& source_strides,
                          int64_t count, size_t element_size) {
    if (count == 0) return;
    int64_t block = 1;
    size_t outer_rank = shape.size();
    while (outer_rank > 0) {
        const size_t d = outer_rank - 1;
        if (shape[d] != 1 && source_strides[d] != block) break;
        block *= shape[d];
        --outer_rank;
    }
    const auto* src = static_cast<const char*>(source);
    auto* dst = static_cast<char*>(destination);
    const size_t block_bytes = static_cast<size_t>(block) * element_size;
    std::vector<int64_t> coordinate(outer_rank, 0);
    int64_t offset = 0;
    for (int64_t out = 0; out < count; out += block) {
        std::memcpy(dst + static_cast<size_t>(out) * element_size,
                    src + static_cast<size_t>(offset) * element_size, block_bytes);
        for (size_t d = outer_rank; d-- > 0;) {
            if (++coordinate[d] < shape[d]) {
                offset += source_strides[d];
                break;
            }
            coordinate[d] = 0;
            offset -= (shape[d] - 1) * source_strides[d];
        }
    }
}

} // namespace detail
} // namespace ops
