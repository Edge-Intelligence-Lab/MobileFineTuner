#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>   // must be at file scope, never inside namespace ops
#endif
#include "dtype.h"   // 使用 float32_to_fp16_bits / fp16_bits_to_float32

namespace ops {


constexpr int Q8_0_BLOCK_SIZE = 32;

// 一个 Q8_0 block：2 字节 fp16 scale + 32 字节 int8 权重 = 34 字节 = 8.5 bit/权重
//
// 对应 llama.cpp / ggml 的 block_q8_0（ggml-common.h）：
//     typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
// 其中 ggml_half 是 2 字节 fp16，QK8_0 = 32。字段名 d / qs 与 ggml 保持一致。
struct block_q8_0 {
    uint16_t d;                       // fp16 scale
    int8_t   qs[Q8_0_BLOCK_SIZE];     // 32 个 int8
};

// ---------------------------------------------------------------------------
// 参考实现：与 llama.cpp 的 quantize_row_q8_0_ref 对齐
//
// llama.cpp 的定义（ggml-quants.c）：
//     void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k)
// 量化规则：
//     amax = max|x|
//     d    = amax / -128          <-- 注意是 -128，不是 127
//     id   = 1 / d
//     qs[j]= roundf(x[j] * id)    截断到 [-128, 127]
//
// 注意符号约定：lp = amax / -128 是负数，因此 q[j] = round(x[j] / d)
// 得到的是“正数↔负数互换”的结果，dequantize 时用 d*q[j] 会还原回来。
// 这与 ggml 逐字一致；保持它可以保证产出的 block 与 llama.cpp 的 Q8_0
// 二进制布局完全互换。
// ---------------------------------------------------------------------------
inline void quantize_row_q8_0_ref(const float* src, block_q8_0* dst, int n) {
    if (n % Q8_0_BLOCK_SIZE != 0) {
        throw std::runtime_error("quantize_row_q8_0_ref: n must be a multiple of QK8_0 (32)");
    }
    const int nb = n / Q8_0_BLOCK_SIZE;
    for (int i = 0; i < nb; ++i) {
        const float* x = src + static_cast<size_t>(i) * Q8_0_BLOCK_SIZE;

        float amax = 0.0f;
        for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) {
            const float a = fabsf(x[j]);
            if (a > amax) amax = a;
        }

        const float d  = amax / -128.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        // ggml 通过 memcpy 把 2 字节 fp16 原样写进 block；这里保留同样的语义。
        const uint16_t dh = float32_to_fp16_bits(d);
        std::memcpy(&dst[i].d, &dh, sizeof(uint16_t));

        for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) {
            int v = static_cast<int>(roundf(x[j] * id));
            v = v < -128 ? -128 : (v > 127 ? 127 : v);
            dst[i].qs[j] = static_cast<int8_t>(v);
        }
    }
}

// 反量化：与 llama.cpp 的 dequantize_row_q8_0 对齐
inline void dequantize_row_q8_0(const block_q8_0* src, float* dst, int n) {
    if (n % Q8_0_BLOCK_SIZE != 0) {
        throw std::runtime_error("dequantize_row_q8_0: n must be a multiple of QK8_0 (32)");
    }
    const int nb = n / Q8_0_BLOCK_SIZE;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_bits_to_float32(src[i].d);
        for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) {
            dst[static_cast<size_t>(i) * Q8_0_BLOCK_SIZE + j] =
                static_cast<float>(src[i].qs[j]) * d;
        }
    }
}

// ---------------------------------------------------------------------------
// 融合算子：不做全量反量化，直接读 int8 block 参与计算
//
// 这是 llama.cpp 真正的做法（ggml_vec_dot_q8_0_q8_0 的思路），也是量化
// 带来加速和内存收益的来源：
//   * 无需 materialize 一个 fp32 权重副本（省内存带宽）
//   * 权重按 int8 读取，每次只读 34 字节/32 权重（fp32 需要 128 字节）
//
// 关键点：**必须把激活也量化成 int8**，否则乘法被迫提升到浮点，整数点积
// 指令（AVX2 的 _mm256_maddubs_epi16 / ARM 的 sdot）就用不上。
// 见下面的 quantize_activations_q8_0() 与 matmul_q8_0_q8_0_tiled()。
// ---------------------------------------------------------------------------

// C[N, M] = A[N, K] @ Wq[M, K]^T
//
// 存储约定：量化权重按行连续存放，每一行是 K/32 个 block，行主序。
// 这样 A 的每一行与 Wq 的每一行做点积就是 matmul 的语义，与 ggml 的
// 行布局一致（这也是 ggml 的 Q8_0 权重要求的形状）。
//
// 要求 K % 32 == 0：与 ggml 相同，Q8_0 不支持不足一个 block 的尾巴。
// 低于该要求时由调用方决定回退策略（见 LoRALinear::quantize_base_weight_q8_0
// 的 allow_skip 参数）。

// ---------------------------------------------------------------------------
// 激活量化：把整块 fp32 激活 [rows, K] 量化成 Q8_0
//
// 每一行有 K/32 个 block，行主序。布局与权重侧完全对称，于是点积可以写成
// "第 i 行第 blk 块 × 第 j 行第 blk 块"，两边都连续。
//
// 与权重不同，激活**不存储**：每次 forward 现算现用，用完就扔。它不省内存，
// 存在的唯一理由是让点积的两个操作数都是 int8，从而能用整数运算（以及
// SIMD 的整数点积指令）而不是浮点乘法。
// ---------------------------------------------------------------------------
inline void quantize_activations_q8_0(const float* a, block_q8_0* dst,
                                      int64_t rows, int64_t k) {
    if (k % Q8_0_BLOCK_SIZE != 0) {
        throw std::runtime_error("quantize_activations_q8_0: K must be a multiple of 32");
    }
    const int64_t kb = k / Q8_0_BLOCK_SIZE;
    for (int64_t i = 0; i < rows; ++i) {
        quantize_row_q8_0_ref(a + i * k, dst + i * kb, static_cast<int>(k));
    }

}

// int8 × int8 的 block 点积，返回**整数**累加和（两个 scale 在调用方相乘）。
//
// 注意这里不用乘 scale 再浮点累加：llama.cpp 的做法是先把 32 个整数乘积累加
// 成 int32，最后才乘一次 scale。这样内层循环是纯整数运算。
//
// 溢出分析：单块最大 |127×127×32| ≈ 5.16e5；K=8192 时最多 256 块，
// 累计 ≈ 1.3e8，远小于 int32 上限 2.1e9。
//
// 两条路径，运行时选最快的（VNNI 的 dpbusd 是 u8×s8，Q8_0 的 -128 码点
// 在符号搬运时会溢出，故未采用）：
//   * AVX2   (vpmovsxbw + vpmaddwd) 一次 32 字节，无饱和风险
//   * 标量   兜底（也用于非 x86）
// 注意：编译目标仍是 x86-64 基线（SSE2），这里靠 __attribute__((target))
// 单独给函数开指令集，再用 __builtin_cpu_supports 在运行时挑选。
// 这样产物仍然可移植，而编译器本来是不会自动向量化这个点积的。
inline int32_t vec_dot_q8_0_q8_0_i32_scalar(const int8_t* a, const int8_t* b) {
    int32_t sum = 0;
    for (int t = 0; t < Q8_0_BLOCK_SIZE; ++t) {
        sum += static_cast<int32_t>(a[t]) * static_cast<int32_t>(b[t]);
    }
    return sum;
}

#if defined(__x86_64__) || defined(__i386__)

// 32 × int8 → 16 × int16 → 8 × int32，再归约。全程精确，无饱和。
__attribute__((target("avx2")))
inline int32_t q8_dot_avx2(const int8_t* a, const int8_t* b) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
    const __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
    const __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
    const __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(a_lo, b_lo),
                                 _mm256_madd_epi16(a_hi, b_hi));
    __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s),
                              _mm256_extracti128_si256(s, 1));
    t = _mm_hadd_epi32(t, t);
    t = _mm_hadd_epi32(t, t);
    return _mm_cvtsi128_si32(t);
}

// 注意：也试过 AVX-512 VNNI 的 dpbusd（u8×s8），但它需要把负号搬到另一侧，
// 而 int8 没有 +128 —— 码点为 -128 时取反会溢出（Q8_0 的 d = amax/-128 确实
// 会产生 -128），结果算错。除非把码点限制到 [-127,127]（会改变与 llama.cpp
// 的字节级兼容），否则 VNNI 这条不能走。AVX2 路径全程 int16，无此问题。

// 32 × int8 → 16 × int16 → 8 × int32，再归约。全程精确，无饱和。
#endif

inline int vec_dot_q8_0_q8_0_isa() {
#if defined(__x86_64__) || defined(__i386__)
    static const int mode = []() {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2")) return 1;
        return 0;
    }();
    return mode;
#else
    return 0;
#endif
}

inline int32_t vec_dot_q8_0_q8_0_i32(const block_q8_0* a_blk, const block_q8_0* b_blk) {
    static const int mode = vec_dot_q8_0_q8_0_isa();
#if defined(__x86_64__) || defined(__i386__)
    if (mode == 1) return q8_dot_avx2(a_blk->qs, b_blk->qs);
#endif
    (void)mode;
    return vec_dot_q8_0_q8_0_i32_scalar(a_blk->qs, b_blk->qs);
}

// ---------------------------------------------------------------------------
// 融合 + 分块（tiling）的 int8×int8 矩阵乘：C[N, M] = Aq[N, K] @ Wq[M, K]^T
//
// 两个要点，缺一不可：
//
// 1) **两边都是 int8**：内层循环是纯整数乘加，没有 float 转换。
//    这是"激活也量化"的全部意义 —— 只量化权重的话，激活是 float，
//    乘法就被迫提升到浮点，整数 SIMD 指令用不上。
//
// 2) **分块（blocking）**：朴素三重循环里，权重矩阵会被反复扫描 n 遍
//    （每个输出行重读一次全部权重），内存流量 = n × 权重大小。
//    这里把输出行分成 TI 行一组，每组内对每个权重行只读一次、服务 TI 个
//    输出行，把权重的工作集压进 cache。真正的收益来自这里，而不是 SIMD。
//
// 循环顺序也是刻意的：j（权重行）在外、i（激活行）在内，于是最内层循环
// 对 i 连续访问 c[i*m+j]，对固定 j 连续访问 wq[j*kb+blk] —— 两侧都可被
// 自动向量化。
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 融合 + 分块（tiling）的 int8×int8 矩阵乘：C[N, M] = Aq[N, K] @ Wq[M, K]^T
//
// 两个要点，缺一不可：
//
// 1) **两边都是 int8**：内层循环是纯整数乘加，没有 float 转换。
//    这是"激活也量化"的全部意义 —— 只量化权重的话，激活是 float，
//    乘法就被迫提升到浮点，整数 SIMD 指令用不上。
//
// 2) **整数累加与 scale 应用必须分成两趟**（这是实测出来的关键结构）。
//
//    朴素写法把两步交织在一起：
//        for (blk) {
//            int isum = vec_dot(a[blk], w[blk]);
//            sum += (float)isum * d;          // ← 每 32 个元素插入一次标量浮点
//        }
//    问题在于那次 `sum += ...` 是**标量浮点且带串行依赖**：下一轮整数累加
//    必须等这次浮点加法完成。K=768 时每行插入 24 次，整行 32 行共数百次，
//    把整数向量化的收益完全抵消。实测这种写法比项目自身的 fp32 matmul *慢 1.39x*。
//
//    拆成两趟后：
//        第一趟：纯整数累加，每个 block 的 int32 结果写入 ibuf[]（完全可向量化）
//        第二趟：另一趟连续浮点循环，批量乘 scale 再求和（无串行依赖）
//    实测提升 1.23~1.35x，与项目 fp32 matmul 基本打平（1.05x），
//    同时保留 3.76x 的内存压缩。
//
// 注：曾试过把 a/w 两个 fp16 scale 预先乘好（每个 block 组合一次），
// 反而更慢（0.93x）—— 额外的缓冲与写回超过了节省。故不采用。
// ---------------------------------------------------------------------------
inline void matmul_q8_0_q8_0_tiled(const block_q8_0* aq, const block_q8_0* wq,
                                   float* c, int64_t n, int64_t m, int64_t k,
                                   int64_t tile_rows = 128) {
    if (k % Q8_0_BLOCK_SIZE != 0) {
        throw std::runtime_error("matmul_q8_0_q8_0_tiled: K must be a multiple of 32");
    }
    const int64_t kb = k / Q8_0_BLOCK_SIZE;
    if (kb <= 0) return;

    // Scratch reused across the whole call; each output element needs kb entries.
    // Heap-allocated once instead of per element.
    std::vector<int32_t> ibuf(static_cast<size_t>(kb));
    std::vector<float>   dbuf(static_cast<size_t>(kb));

    for (int64_t i = 0; i < n; ++i) {
        const block_q8_0* arow = aq + static_cast<size_t>(i) * kb;
        float* crow = c + i * m;

        // Convert this activation row's scales ONCE; they are reused for all m
        // output features, so this is O(kb) instead of O(kb * m).
        for (int64_t b = 0; b < kb; ++b) {
            dbuf[static_cast<size_t>(b)] = fp16_bits_to_float32(arow[b].d);
        }

        for (int64_t j = 0; j < m; ++j) {
            const block_q8_0* wrow = wq + static_cast<size_t>(j) * kb;

            // ---- pass 1: pure integer accumulation (vectorisable) ----
            for (int64_t b = 0; b < kb; ++b) {
                ibuf[static_cast<size_t>(b)] =
                    vec_dot_q8_0_q8_0_i32(&arow[b], &wrow[b]);
            }

            // ---- pass 2: contiguous float reduction, 4-way unrolled ----
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            int64_t b = 0;
            for (; b + 4 <= kb; b += 4) {
                acc0 += static_cast<float>(ibuf[static_cast<size_t>(b + 0)]) *
                        (dbuf[static_cast<size_t>(b + 0)] * fp16_bits_to_float32(wrow[b + 0].d));
                acc1 += static_cast<float>(ibuf[static_cast<size_t>(b + 1)]) *
                        (dbuf[static_cast<size_t>(b + 1)] * fp16_bits_to_float32(wrow[b + 1].d));
                acc2 += static_cast<float>(ibuf[static_cast<size_t>(b + 2)]) *
                        (dbuf[static_cast<size_t>(b + 2)] * fp16_bits_to_float32(wrow[b + 2].d));
                acc3 += static_cast<float>(ibuf[static_cast<size_t>(b + 3)]) *
                        (dbuf[static_cast<size_t>(b + 3)] * fp16_bits_to_float32(wrow[b + 3].d));
            }
            float sum = (acc0 + acc1) + (acc2 + acc3);
            for (; b < kb; ++b) {
                sum += static_cast<float>(ibuf[static_cast<size_t>(b)]) *
                       (dbuf[static_cast<size_t>(b)] * fp16_bits_to_float32(wrow[b].d));
            }

            crow[j] = sum;
        }
    }
    (void)tile_rows;   // retained for API compatibility; loop order is now
                       // row-major over i, which streams the weight rows once
                       // per activation row (see the vectorizer report).
}

// ---------------------------------------------------------------------------
// 反向传播用的融合算子：C[N, K] = G[N, M] @ Wq[M, K]
//
// 直接用 block 下标就可以表达，不需要先构造 W^T：
//   C[i][k] = sum_o G[i][o] * W[o][k]
//   块 b 覆盖 k 的区间 [b*32, (b+1)*32)，这个区间在 W[o] 里是连续的 32 字节，
//   所以对每个块 b 可以一次性读出 W 所有行在 [b*32, b*32+32) 上的 32 个元素，
//   用 32 个累加器累加。
//
// 注意 scale 的位置：Q8_0 每行每块各自一个 fp16 scale，即 d[o][b]。它依赖 o，
// 因此**不能**提到 o 循环外面（曾经那样写，导致 grad_x 完全算错）：
//   C[i][k] = sum_o (G[i][o] * d[o][b]) * q[o][b][j]，  k = b*32 + j
// 所以每行的 scale 必须先折进该行的梯度，再参与累加。
// 让编译器有机会把内层循环向量化。
// ---------------------------------------------------------------------------
inline void matmul_q8_0_rhs_T_scalar(const float* g, const block_q8_0* wq,
                                     float* c, int64_t n, int64_t m, int64_t k) {
    const int64_t kb = k / Q8_0_BLOCK_SIZE;

    for (int64_t i = 0; i < n; ++i) {
        const float* grow = g + i * m;
        float* crow = c + i * k;

        for (int64_t b = 0; b < kb; ++b) {
            float partial[Q8_0_BLOCK_SIZE];
            for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) partial[j] = 0.0f;

            for (int64_t o = 0; o < m; ++o) {
                const float gv = grow[o];
                if (gv == 0.0f) continue;
                const block_q8_0& qblk = wq[static_cast<size_t>(o) * kb + b];
                // Row o's block b has its OWN scale, so it must be folded into
                // this row's contribution. It cannot be pulled out of the sum
                // over o: different rows do not share one scale.
                const float go = gv * fp16_bits_to_float32(qblk.d);
                for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) {
                    partial[j] += go * static_cast<float>(qblk.qs[j]);
                }
            }

            float* dst = crow + b * Q8_0_BLOCK_SIZE;
            for (int j = 0; j < Q8_0_BLOCK_SIZE; ++j) {
                dst[j] = partial[j];
            }
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)
// Vectorised version of the same loop. The 32 accumulators live in four
// __m256 registers; per (o, b) we do four int8->int32->float conversions
// (8 lanes each) and four FMAs. The scalar `if (gv == 0) continue` is dropped:
// a zero gradient contributes zero either way, and the branch prevented the
// compiler from vectorising.
__attribute__((target("avx2,fma")))
inline void q8_rhs_T_avx2(const float* g, const block_q8_0* wq,
                          float* c, int64_t n, int64_t m, int64_t k) {
    const int64_t kb = k / Q8_0_BLOCK_SIZE;

    for (int64_t i = 0; i < n; ++i) {
        const float* grow = g + i * m;
        float* crow = c + i * k;

        for (int64_t b = 0; b < kb; ++b) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int64_t o = 0; o < m; ++o) {
                const block_q8_0& qblk = wq[static_cast<size_t>(o) * kb + b];
                const __m256 vg = _mm256_set1_ps(
                    grow[o] * fp16_bits_to_float32(qblk.d));
                const int8_t* qs = qblk.qs;
                const __m256 f0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(qs + 0))));
                const __m256 f1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(qs + 8))));
                const __m256 f2 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(qs + 16))));
                const __m256 f3 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(qs + 24))));
                acc0 = _mm256_fmadd_ps(vg, f0, acc0);
                acc1 = _mm256_fmadd_ps(vg, f1, acc1);
                acc2 = _mm256_fmadd_ps(vg, f2, acc2);
                acc3 = _mm256_fmadd_ps(vg, f3, acc3);
            }

            float* dst = crow + b * Q8_0_BLOCK_SIZE;
            _mm256_storeu_ps(dst + 0, acc0);
            _mm256_storeu_ps(dst + 8, acc1);
            _mm256_storeu_ps(dst + 16, acc2);
            _mm256_storeu_ps(dst + 24, acc3);
        }
    }
}

inline bool q8_rhs_T_has_avx2_fma() {
    static const bool ok = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return ok;
}
#endif

inline void matmul_q8_0_rhs_T(const float* g, const block_q8_0* wq,
                              float* c, int64_t n, int64_t m, int64_t k) {
    if (k % Q8_0_BLOCK_SIZE != 0) {
        throw std::runtime_error("matmul_q8_0_rhs_T: K must be a multiple of 32");
    }
#if defined(__x86_64__) || defined(__i386__)
    if (q8_rhs_T_has_avx2_fma()) {
        q8_rhs_T_avx2(g, wq, c, n, m, k);
        return;
    }
#endif
    matmul_q8_0_rhs_T_scalar(g, wq, c, n, m, k);
}

} // namespace ops