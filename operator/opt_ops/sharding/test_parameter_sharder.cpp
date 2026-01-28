/**
 * @file test_parameter_sharder.cpp
 * @brief Sanity test for ParameterSharder: register -> offload -> reload -> verify.
 *
 * Build example:
 *   g++ -std=c++17 -I. \
 *       opt_ops/sharding/parameter_sharder.cpp \
 *       opt_ops/sharding/test_parameter_sharder.cpp \
 *       finetune_ops/core/tensor.cpp \
 *       finetune_ops/core/ops.cpp \
 *       finetune_ops/core/backward_functions.cpp \
 *       finetune_ops/core/autograd_engine.cpp \
 *       finetune_ops/core/logger.cpp \
 *       finetune_ops/core/memory_manager.cpp \
 *       finetune_ops/core/step_arena.cpp \
 *       finetune_ops/core/utils.cpp \
 *       finetune_ops/core/memory_efficient_attention.cpp \
 *       -pthread -o /tmp/test_param_sharder
 */

#include "parameter_sharder.h"
#include <iostream>
#include <random>
#include <cassert>

using namespace ops;
using namespace ops::sharding;

static TensorPtr make_tensor(int64_t rows, int64_t cols, float seed) {
    TensorPtr t = std::make_shared<Tensor>(std::vector<int64_t>{rows, cols}, kFloat32, kCPU);
    float* data = t->data<float>();
    for (int64_t i = 0; i < rows * cols; ++i) {
        data[i] = seed + static_cast<float>(i) * 0.001f;
    }
    return t;
}

static bool tensor_equal(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b) return false;
    if (a->shape() != b->shape()) return false;
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    // Allow slight error after FP16 disk quantization/dequantization
    const float tol = 2e-3f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        if (std::abs(pa[i] - pb[i]) > tol) return false;
    }
    return true;
}

int main() {
    ShardConfig cfg;
    cfg.offload_dir = "/tmp/param_shard_test";
    cfg.max_resident_bytes = 1024 * 16; // 16 KB budget to force eviction
    cfg.quantize_fp16_on_disk = true;
    
    ParameterSharder sharder(cfg);
    
    auto t1 = make_tensor(64, 4, 0.1f);  // ~1KB
    auto t2 = make_tensor(64, 4, 1.0f);
    auto t3 = make_tensor(64, 4, 2.0f);
    
    sharder.register_parameter("layer0.w", t1, true);
    sharder.register_parameter("layer1.w", t2, true);
    sharder.register_parameter("layer2.w", t3, true);
    
    // Trigger budget: requesting layer2 will evict earlier ones
    auto r2 = sharder.require("layer2.w");
    auto r0 = sharder.require("layer0.w"); // Will evict layer1
    
    // Modify and mark dirty to ensure rewrite to disk
    r0->data<float>()[0] += 1.0f;
    sharder.mark_dirty("layer0.w");
    
    // Force offload all, then reload and verify consistency
    sharder.offload_all();
    auto reload0 = sharder.require("layer0.w");
    auto reload2 = sharder.require("layer2.w");
    
    assert(reload0->data<float>()[0] - t1->data<float>()[0] - 1.0f < 1e-4);
    assert(tensor_equal(reload2, t3));
    
    std::cout << "[OK] ParameterSharder offload/reload sanity passed.\n";
    std::cout << sharder.debug_string() << std::endl;
    return 0;
}
