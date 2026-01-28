#include "parameter_sharder.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <cstring>

using namespace ops;
using namespace ops::sharding;

// Simple test helper to create a tensor filled with random data
TensorPtr create_random_tensor(const std::vector<int64_t>& shape) {
    auto t = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    float* data = t->data<float>();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < t->numel(); ++i) {
        data[i] = dist(rng);
    }
    return t;
}

void test_strict_budget_lru() {
    std::cout << "[Test] Strict Budget & LRU Eviction... ";
    
    // Config: 15MB budget. 
    // We will create 10MB tensors (2.5M floats).
    size_t param_size = 2500000; 
    size_t param_bytes = param_size * sizeof(float); // 10MB
    
    ShardConfig cfg;
    cfg.offload_dir = "/tmp/test_sharder_strict";
    cfg.max_resident_bytes = 15 * 1024 * 1024; // 15MB
    cfg.quantize_fp16_on_disk = false; // Test logic first without quantization noise
    
    ParameterSharder sharder(cfg);
    
    TensorPtr tA = create_random_tensor({2500000}); // 10MB
    TensorPtr tB = create_random_tensor({2500000}); // 10MB
    TensorPtr tC = create_random_tensor({2500000}); // 10MB
    
    TensorPtr refA = nullptr, refB = nullptr, refC = nullptr;
    
    // Register A. Should stay in memory (10MB < 15MB).
    sharder.register_parameter("A", tA, true, &refA);
    assert(refA != nullptr);
    assert(refA == tA);
    
    // Register B. Total needed 20MB > 15MB. A is LRU. Should evict A.
    // NOTE: register_parameter with keep_in_memory=true triggers load_entry which triggers ensure_budget.
    sharder.register_parameter("B", tB, true, &refB);
    
    assert(refB != nullptr);
    assert(refB == tB);
    
    // A should be evicted
    if (refA != nullptr) {
        std::cerr << "FAILED: A should be evicted (refA should be null)\n";
        exit(1);
    }
    
    // Require A. Should evict B.
    auto loadedA = sharder.require("A");
    assert(loadedA != nullptr);
    assert(refA == loadedA);
    
    if (refB != nullptr) {
        std::cerr << "FAILED: B should be evicted when loading A\n";
        exit(1);
    }
    
    // Register C. Should evict A.
    sharder.register_parameter("C", tC, true, &refC);
    assert(refC != nullptr);
    if (refA != nullptr) {
        std::cerr << "FAILED: A should be evicted when registering C\n";
        exit(1);
    }
    
    std::cout << "OK\n";
}

void test_data_integrity_fp16() {
    std::cout << "[Test] Data Integrity (FP16 quantization)... ";
    
    ShardConfig cfg;
    cfg.offload_dir = "/tmp/test_sharder_strict";
    cfg.max_resident_bytes = 100 * 1024 * 1024; // Plenty of space
    cfg.quantize_fp16_on_disk = true; // Enable quantization
    
    ParameterSharder sharder(cfg);
    
    // Create tensor with known pattern
    int64_t size = 1000;
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{size}, kFloat32, kCPU);
    float* data = t->data<float>();
    for (int i = 0; i < size; ++i) data[i] = (float)i * 0.001f; // 0.000, 0.001, ... 0.999
    
    // Save original data for verification
    std::vector<float> original(data, data + size);
    
    TensorPtr ref = nullptr;
    sharder.register_parameter("integrity_test", t, true, &ref);
    
    // Force offload
    sharder.offload_all();
    assert(ref == nullptr);
    
    // Reload
    auto loaded = sharder.require("integrity_test");
    assert(ref == loaded);
    
    // Verify
    const float* new_data = loaded->data<float>();
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(new_data[i] - original[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    // FP16 precision is roughly 1e-3 to 1e-4 for this range
    if (max_diff > 1e-3) {
        std::cerr << "FAILED: Max diff " << max_diff << " exceeds tolerance\n";
        exit(1);
    }
    
    std::cout << "OK (Max diff: " << max_diff << ")\n";
}

int main() {
    // Clean up temp dir
    system("rm -rf /tmp/test_sharder_strict");
    
    test_strict_budget_lru();
    test_data_integrity_fp16();
    
    std::cout << "All Strict Tests Passed.\n";
    return 0;
}

