#include "quant_io.h"

#include <cstdio>
#include <cstring>

namespace ops {
namespace {

constexpr char kMagic[8] = {'M', 'F', 'T', 'Q', '8', '\0', '\0', '\0'};
constexpr uint32_t kVersion = 1;

// Q8_0 blocks are copied verbatim, so the on-disk size must match the struct.
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes on disk");

bool write_all(std::FILE* f, const void* data, size_t n) {
    return n == 0 || std::fwrite(data, 1, n, f) == n;
}

template <typename T>
bool write_pod(std::FILE* f, const T& v) {
    return write_all(f, &v, sizeof(T));
}

bool read_all(std::FILE* f, void* data, size_t n) {
    return n == 0 || std::fread(data, 1, n, f) == n;
}

template <typename T>
bool read_pod(std::FILE* f, T& v) {
    return read_all(f, &v, sizeof(T));
}

}  // namespace

bool write_quant_model(const std::string& path, const std::vector<QuantTensor>& tensors,
                       std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot open " + path + " for writing";
        return false;
    }

    bool ok = write_all(f, kMagic, sizeof(kMagic)) && write_pod(f, kVersion) &&
              write_pod(f, static_cast<uint32_t>(tensors.size()));

    for (size_t i = 0; ok && i < tensors.size(); ++i) {
        const QuantTensor& t = tensors[i];
        const uint32_t name_len = static_cast<uint32_t>(t.name.size());
        ok = write_pod(f, name_len) && write_all(f, t.name.data(), name_len);

        const uint8_t flag = t.is_q8 ? 1 : 0;
        ok = ok && write_pod(f, flag);

        if (!ok) break;
        if (t.is_q8) {
            const uint64_t nblocks = static_cast<uint64_t>(t.q8.size());
            ok = write_pod(f, static_cast<uint32_t>(t.k)) &&
                 write_pod(f, static_cast<uint32_t>(t.m)) && write_pod(f, nblocks) &&
                 write_all(f, t.q8.data(), nblocks * sizeof(block_q8_0));
        } else {
            const uint32_t ndim = static_cast<uint32_t>(t.shape.size());
            ok = write_pod(f, ndim);
            for (uint32_t d = 0; ok && d < ndim; ++d) ok = write_pod(f, t.shape[d]);
            const uint64_t count = static_cast<uint64_t>(t.f32.size());
            ok = ok && write_pod(f, count) &&
                 write_all(f, t.f32.data(), count * sizeof(float));
        }
    }

    if (std::fclose(f) != 0) ok = false;
    if (!ok && err) *err = "write failed (disk full?)";
    return ok;
}

bool read_quant_model(const std::string& path, std::vector<QuantTensor>& out,
                      std::string* err) {
    out.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }

    char magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    if (!read_all(f, magic, sizeof(magic)) || std::memcmp(magic, kMagic, 8) != 0) {
        std::fclose(f);
        if (err) *err = "bad magic (not a Q8_0 model file)";
        return false;
    }
    if (!read_pod(f, version) || version != kVersion) {
        std::fclose(f);
        if (err) *err = "unsupported version " + std::to_string(version);
        return false;
    }
    if (!read_pod(f, count)) {
        std::fclose(f);
        if (err) *err = "truncated header";
        return false;
    }

    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        QuantTensor t;
        uint32_t name_len = 0;
        if (!read_pod(f, name_len) || name_len > (1u << 20)) {
            if (err) *err = "bad tensor name length";
            std::fclose(f);
            return false;
        }
        t.name.resize(name_len);
        if (!read_all(f, t.name.data(), name_len)) {
            if (err) *err = "truncated tensor name";
            std::fclose(f);
            return false;
        }

        uint8_t flag = 0;
        if (!read_pod(f, flag)) {
            if (err) *err = "truncated tensor flag";
            std::fclose(f);
            return false;
        }
        t.is_q8 = (flag != 0);

        if (t.is_q8) {
            uint32_t k = 0, m = 0;
            uint64_t nblocks = 0;
            if (!read_pod(f, k) || !read_pod(f, m) || !read_pod(f, nblocks)) {
                if (err) *err = "truncated q8 header";
                std::fclose(f);
                return false;
            }
            if (nblocks == 0 || nblocks > (1ull << 32)) {
                if (err) *err = "implausible block count";
                std::fclose(f);
                return false;
            }
            t.k = k;
            t.m = m;
            t.q8.resize(static_cast<size_t>(nblocks));
            if (!read_all(f, t.q8.data(), nblocks * sizeof(block_q8_0))) {
                if (err) *err = "truncated q8 payload";
                std::fclose(f);
                return false;
            }
        } else {
            uint32_t ndim = 0;
            if (!read_pod(f, ndim) || ndim == 0 || ndim > 8) {
                if (err) *err = "bad ndim";
                std::fclose(f);
                return false;
            }
            t.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                if (!read_pod(f, t.shape[d])) {
                    if (err) *err = "truncated shape";
                    std::fclose(f);
                    return false;
                }
            }
            uint64_t n = 0;
            if (!read_pod(f, n)) {
                if (err) *err = "truncated payload size";
                std::fclose(f);
                return false;
            }
            t.f32.resize(static_cast<size_t>(n));
            if (!read_all(f, t.f32.data(), n * sizeof(float))) {
                if (err) *err = "truncated fp32 payload";
                std::fclose(f);
                return false;
            }
        }
        out.push_back(std::move(t));
    }

    std::fclose(f);
    return true;
}

}  // namespace ops
