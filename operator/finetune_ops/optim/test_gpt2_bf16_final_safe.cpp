#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include "../core/dtype.h"
#include "adam.h"

using namespace ops;

static long long mem_available_kb() {
    std::ifstream f("/proc/meminfo");
    std::string key, unit;
    long long value;

    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:")
            return value;
    }

    return -1;
}

static void print_mem(const char* phase) {
    const auto kb = mem_available_kb();
    std::cout << "[MEM] " << phase << ": "
              << (kb / 1024.0) << " MiB available\n";
}

static bool finite_tensor(const TensorPtr& t) {
    if (!t || t->dtype() != kFloat32)
        return false;

    const float* p = t->data<float>();

    for (int64_t i = 0; i < t->numel(); ++i) {
        if (!std::isfinite(p[i]))
            return false;
    }

    return true;
}

int main() {
    constexpr long long HARD_STOP_KB = 600LL * 1024LL;
    constexpr long long ADAM_MIN_KB = 1500LL * 1024LL;

    std::atomic<bool> stop_watchdog{false};

    // Hard memory watchdog.
    std::thread watchdog([&]() {
        while (!stop_watchdog.load()) {
            const long long kb = mem_available_kb();

            if (kb > 0 && kb < HARD_STOP_KB) {
                std::cerr
                    << "\n[SAFE ABORT] MemAvailable dropped below 600 MiB.\n";

                std::_Exit(99);
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(250));
        }
    });

    try {
        const std::string model_dir = "/root/gpt2-tamil-124m";

        std::cout << "============================================\n";
        std::cout << " FINAL 124M BF16 WEIGHT TEST\n";
        std::cout << "============================================\n";

        print_mem("start");

        // Require a reasonable starting margin.
        if (mem_available_kb() < 1800LL * 1024LL) {
            std::cout << "[SAFE STOP] Less than 1.8 GiB available.\n";
            stop_watchdog = true;
            watchdog.join();
            return 0;
        }

        // ---------- MODEL ----------
        GPT2Config config = GPT2Config::from_pretrained(model_dir);
        GPT2Model model(config);

        if (config.tie_word_embeddings)
            model.tie_weights();

        // Load real checkpoint as BF16.
        SafeTensorsModelReader reader(model_dir);
        reader.parse_headers();

        auto mapping =
            GPT2KeyMapper::generate_gpt2_mapping(config.n_layer);

        SafeTensorsLoadOptions options;
        options.transpose_linear = false;
        options.auto_promote_fp16 = true;
        options.convert_f32_to_bf16 = true;
        options.verbose = false;

        auto tensors =
            reader.load_tensors_mapped(mapping, options);

        size_t total_params = 0;
        size_t bf16_tensors = 0;
        size_t fp32_tensors = 0;

        for (const auto& kv : tensors) {
            if (!kv.second)
                continue;

            total_params += static_cast<size_t>(kv.second->numel());

            if (kv.second->dtype() == kBFloat16)
                ++bf16_tensors;
            else if (kv.second->dtype() == kFloat32)
                ++fp32_tensors;

            model.assign_weight(kv.first, kv.second);
        }

        std::cout << "Parameters:    " << total_params << "\n";
        std::cout << "BF16 tensors:  " << bf16_tensors << "\n";
        std::cout << "FP32 tensors:  " << fp32_tensors << "\n";

        if (total_params != 124439808 ||
            bf16_tensors != 148 ||
            fp32_tensors != 0) {
            std::cerr << "[FAIL] BF16 weight verification failed.\n";
            stop_watchdog = true;
            watchdog.join();
            return 2;
        }

        std::cout << "[PASS] All 124M model parameters are BF16.\n";
        print_mem("after BF16 load");

        // ---------- ENABLE TRAINING ----------
        for (auto& p : model.parameters()) {
            if (p)
                p->set_requires_grad(true);
        }

        // Tiny real input: 8 known Tamil GPT-2 token IDs.
        const int64_t ids[8] = {
            288, 275, 271, 265,
            281, 262, 742, 302
        };

        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 8},
            ids,
            kInt64,
            kCPU);

        std::cout << "\n===== FORWARD =====\n";
        print_mem("before forward");

        auto logits = model.forward(input_ids);

        if (!logits ||
            logits->dtype() != kFloat32 ||
            logits->shape().size() != 3 ||
            logits->shape()[0] != 1 ||
            logits->shape()[1] != 8 ||
            !finite_tensor(logits)) {
            std::cerr << "[FAIL] Forward produced invalid logits.\n";
            stop_watchdog = true;
            watchdog.join();
            return 3;
        }

        std::cout << "Logits shape: ["
                  << logits->shape()[0] << ","
                  << logits->shape()[1] << ","
                  << logits->shape()[2] << "]\n";

        std::cout << "[PASS] 124M BF16-weight forward.\n";
        print_mem("after forward");

        if (mem_available_kb() < 1000LL * 1024LL) {
            std::cout
                << "[SAFE STOP] Less than 1 GiB remains before backward.\n";
            stop_watchdog = true;
            watchdog.join();
            return 0;
        }

        // ---------- LOSS ----------
        auto target = std::make_shared<Tensor>(
            logits->shape(),
            kFloat32,
            kCPU);

        float* target_data = target->data<float>();

        for (int64_t i = 0; i < target->numel(); ++i)
            target_data[i] = 0.0f;

        auto loss = mse_loss(logits, target, "mean");

        if (!loss || !std::isfinite(loss->item())) {
            std::cerr << "[FAIL] Loss is invalid.\n";
            stop_watchdog = true;
            watchdog.join();
            return 4;
        }

        std::cout << "Loss: " << loss->item() << "\n";

        // ---------- BACKWARD ----------
        std::cout << "\n===== BACKWARD =====\n";
        print_mem("before backward");

        if (mem_available_kb() < 900LL * 1024LL) {
            std::cout
                << "[SAFE STOP] Less than 900 MiB remains before backward.\n";
            stop_watchdog = true;
            watchdog.join();
            return 0;
        }

        loss->backward();

        size_t grad_count = 0;
        size_t bad_grad_count = 0;

        for (const auto& p : model.parameters()) {
            if (!p)
                continue;

            auto g = p->grad();

            if (g) {
                ++grad_count;

                if (g->dtype() != kFloat32 ||
                    !finite_tensor(g)) {
                    ++bad_grad_count;
                }
            }
        }

        std::cout << "Gradients present: " << grad_count << "\n";
        std::cout << "Invalid gradients: " << bad_grad_count << "\n";

        print_mem("after backward");

        if (grad_count == 0 || bad_grad_count != 0) {
            std::cerr << "[FAIL] Backward gradient validation failed.\n";
            stop_watchdog = true;
            watchdog.join();
            return 5;
        }

        std::cout << "[PASS] 124M backward with BF16 weights.\n";

        // ---------- OPTIONAL ONE ADAM STEP ----------
        std::cout << "\n===== ONE ADAM STEP =====\n";

        const long long before_adam = mem_available_kb();

        std::cout << "MemAvailable before Adam: "
                  << before_adam / 1024.0 << " MiB\n";

        if (before_adam < ADAM_MIN_KB) {
            std::cout
                << "[SAFE STOP] Not enough RAM margin for full FP32 Adam state.\n";
            std::cout
                << "[RESULT] Forward + backward passed; Adam step skipped safely.\n";

            stop_watchdog = true;
            watchdog.join();
            return 0;
        }

        std::vector<TensorPtr> params;
        std::vector<TensorPtr> grads;

        for (const auto& p : model.parameters()) {
            if (!p)
                continue;

            auto g = p->grad();

            if (g) {
                params.push_back(p);
                grads.push_back(g);
            }
        }

        AdamConfig adam_cfg;
        adam_cfg.learning_rate = 1e-5f;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.999f;
        adam_cfg.epsilon = 1e-8f;
        adam_cfg.weight_decay = 0.0f;
        adam_cfg.amsgrad = false;

        Adam adam(adam_cfg);

        adam.step(params, grads);

        print_mem("after Adam");

        // Verify parameters remain BF16 and finite.
        size_t still_bf16 = 0;
        size_t invalid_params = 0;

        for (const auto& p : params) {
            if (p->dtype() == kBFloat16) {
                ++still_bf16;

                const uint16_t* d = p->data<uint16_t>();

                for (int64_t j = 0; j < p->numel(); ++j) {
                    float v = bf16_bits_to_float32(d[j]);

                    if (!std::isfinite(v)) {
                        ++invalid_params;
                        break;
                    }
                }
            }
        }

        std::cout << "BF16 parameters after Adam: "
                  << still_bf16 << "\n";
        std::cout << "Invalid parameter tensors: "
                  << invalid_params << "\n";

        if (still_bf16 != params.size() ||
            invalid_params != 0) {
            std::cerr << "[FAIL] BF16 parameters became invalid after Adam.\n";
            stop_watchdog = true;
            watchdog.join();
            return 6;
        }

        std::cout << "[PASS] One full FP32-Adam update completed.\n";
        std::cout << "[PASS] Parameters remain BF16.\n";

        std::cout << "\n============================================\n";
        std::cout << " FINAL RESULT: ALL REQUESTED STAGES PASSED\n";
        std::cout << "============================================\n";

        stop_watchdog = true;
        watchdog.join();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        stop_watchdog = true;
        watchdog.join();
        return 10;
    }
}
