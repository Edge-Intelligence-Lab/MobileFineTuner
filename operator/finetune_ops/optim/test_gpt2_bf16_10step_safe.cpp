#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

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
    long long value = 0;

    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:")
            return value;
    }

    return -1;
}

static void print_mem(const char* label) {
    const long long kb = mem_available_kb();

    std::cout << "[MEM] " << label << ": "
              << kb / 1024.0 << " MiB available\n";
}

static bool finite_f32(const TensorPtr& t) {
    if (!t || t->dtype() != kFloat32)
        return false;

    const float* p = t->data<float>();

    for (int64_t i = 0; i < t->numel(); ++i) {
        if (!std::isfinite(p[i]))
            return false;
    }

    return true;
}

static bool finite_bf16(const TensorPtr& t) {
    if (!t || t->dtype() != kBFloat16)
        return false;

    const uint16_t* p = t->data<uint16_t>();

    for (int64_t i = 0; i < t->numel(); ++i) {
        if (!std::isfinite(bf16_bits_to_float32(p[i])))
            return false;
    }

    return true;
}

int main() {
    constexpr int STEPS = 10;
    constexpr int64_t SEQ = 8;
    constexpr int64_t VOCAB = 50257;

    // Safety thresholds.
    constexpr long long HARD_STOP_KB = 600LL * 1024LL;
    constexpr long long STEP_MIN_KB = 1000LL * 1024LL;
    constexpr long long START_MIN_KB = 1800LL * 1024LL;

    std::atomic<bool> stop_watchdog{false};

    std::thread watchdog([&]() {
        while (!stop_watchdog.load()) {
            const long long kb = mem_available_kb();

            if (kb > 0 && kb < HARD_STOP_KB) {
                std::cerr
                    << "\n[SAFE ABORT] MemAvailable < 600 MiB\n";
                std::_Exit(99);
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(250));
        }
    });

    auto finish = [&](int rc) {
        stop_watchdog = true;
        watchdog.join();
        return rc;
    };

    try {
        const std::string model_dir =
            "/root/gpt2-tamil-124m";

        std::cout
            << "============================================\n"
            << " SAFE 124M BF16 10-STEP FINE-TUNE\n"
            << "============================================\n";

        print_mem("start");

        if (mem_available_kb() < START_MIN_KB) {
            std::cout
                << "[SAFE STOP] Need >= 1.8 GiB at startup.\n";
            return finish(0);
        }

        // --------------------------------------------------------
        // MODEL
        // --------------------------------------------------------

        GPT2Config config =
            GPT2Config::from_pretrained(model_dir);

        GPT2Model model(config);

        if (config.tie_word_embeddings)
            model.tie_weights();

        // --------------------------------------------------------
        // BF16 LOAD
        // --------------------------------------------------------

        SafeTensorsModelReader reader(model_dir);
        reader.parse_headers();

        auto mapping =
            GPT2KeyMapper::generate_gpt2_mapping(
                config.n_layer);

        SafeTensorsLoadOptions options;
        options.transpose_linear = false;
        options.auto_promote_fp16 = true;
        options.convert_f32_to_bf16 = true;
        options.verbose = false;

        auto tensors =
            reader.load_tensors_mapped(
                mapping,
                options);

        size_t total_params = 0;
        size_t bf16_tensors = 0;
        size_t fp32_tensors = 0;

        for (const auto& kv : tensors) {
            if (!kv.second)
                continue;

            total_params +=
                static_cast<size_t>(kv.second->numel());

            if (kv.second->dtype() == kBFloat16)
                ++bf16_tensors;
            else if (kv.second->dtype() == kFloat32)
                ++fp32_tensors;

            model.assign_weight(kv.first, kv.second);
        }

        std::cout
            << "Parameters:   " << total_params << "\n"
            << "BF16 tensors: " << bf16_tensors << "\n"
            << "FP32 tensors: " << fp32_tensors << "\n";

        if (total_params != 124439808 ||
            bf16_tensors != 148 ||
            fp32_tensors != 0) {

            std::cerr
                << "[FAIL] Unexpected parameter dtype state.\n";
            return finish(2);
        }

        std::cout
            << "[PASS] All 124M parameters are BF16.\n";

        print_mem("after BF16 load");

        // --------------------------------------------------------
        // FULL PARAMETER TRAINING
        // --------------------------------------------------------

        for (auto& p : model.parameters()) {
            if (p)
                p->set_requires_grad(true);
        }

        // --------------------------------------------------------
        // INPUT
        // --------------------------------------------------------

        const int64_t ids[SEQ] = {
            288, 275, 271, 265,
            281, 262, 742, 302
        };

        auto input_ids =
            std::make_shared<Tensor>(
                std::vector<int64_t>{1, SEQ},
                ids,
                kInt64,
                kCPU);

        // --------------------------------------------------------
        // TARGET
        // --------------------------------------------------------
        //
        // Keep the existing tiny deterministic MSE objective.
        // This is a training-path stress test, not the final
        // corpus LM objective.
        //

        auto target =
            std::make_shared<Tensor>(
                std::vector<int64_t>{
                    1,
                    SEQ,
                    VOCAB
                },
                kFloat32,
                kCPU);

        float* target_data = target->data<float>();

        std::fill(
            target_data,
            target_data + target->numel(),
            0.0f);

        const int64_t target_ids[SEQ] = {
            275, 271, 265, 281,
            262, 742, 302, 288
        };

        for (int64_t i = 0; i < SEQ; ++i) {
            target_data[
                i * VOCAB + target_ids[i]
            ] = 1.0f;
        }

        // --------------------------------------------------------
        // ADAM
        // --------------------------------------------------------

        AdamConfig adam_cfg;
        adam_cfg.learning_rate = 1e-5f;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.999f;
        adam_cfg.epsilon = 1e-8f;
        adam_cfg.weight_decay = 0.0f;
        adam_cfg.amsgrad = false;

        Adam adam(adam_cfg);

        // --------------------------------------------------------
        // TRAINING LOOP
        // --------------------------------------------------------

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        for (int step = 1; step <= STEPS; ++step) {
            std::cout
                << "\n--------------------------------------------\n"
                << "STEP " << step << "/" << STEPS << "\n"
                << "--------------------------------------------\n";

            print_mem("before step");

            if (mem_available_kb() < STEP_MIN_KB) {
                std::cout
                    << "[SAFE STOP] < 1 GiB before step.\n"
                    << "Completed " << (step - 1)
                    << " steps safely.\n";
                return finish(0);
            }

            // ----------------------------------------------------
            // FORWARD
            // ----------------------------------------------------

            auto logits =
                model.forward(input_ids);

            if (!logits ||
                logits->dtype() != kFloat32 ||
                logits->shape().size() != 3 ||
                logits->shape()[0] != 1 ||
                logits->shape()[1] != SEQ ||
                logits->shape()[2] != VOCAB ||
                !finite_f32(logits)) {

                std::cerr
                    << "[FAIL] Invalid logits at step "
                    << step << "\n";
                return finish(4);
            }

            // ----------------------------------------------------
            // LOSS
            // ----------------------------------------------------

            auto loss =
                mse_loss(
                    logits,
                    target,
                    "mean");

            if (!loss ||
                !std::isfinite(loss->item())) {

                std::cerr
                    << "[FAIL] Invalid loss at step "
                    << step << "\n";
                return finish(5);
            }

            const float loss_value =
                loss->item();

            if (step == 1)
                first_loss = loss_value;

            last_loss = loss_value;

            std::cout
                << "Loss: "
                << loss_value
                << "\n";

            // ----------------------------------------------------
            // BACKWARD
            // ----------------------------------------------------

            if (mem_available_kb() < STEP_MIN_KB) {
                std::cout
                    << "[SAFE STOP] < 1 GiB before backward.\n";
                return finish(0);
            }

            loss->backward();

            std::vector<TensorPtr> params;
            std::vector<TensorPtr> grads;

            size_t grad_count = 0;
            size_t invalid_grads = 0;

            for (const auto& p : model.parameters()) {
                if (!p)
                    continue;

                auto g = p->grad();

                if (!g)
                    continue;

                ++grad_count;

                if (g->dtype() != kFloat32 ||
                    !finite_f32(g)) {

                    ++invalid_grads;
                    continue;
                }

                params.push_back(p);
                grads.push_back(g);
            }

            std::cout
                << "Gradients: "
                << grad_count
                << "\n";

            std::cout
                << "Invalid gradients: "
                << invalid_grads
                << "\n";

            if (grad_count == 0 ||
                invalid_grads != 0) {

                std::cerr
                    << "[FAIL] Gradient validation failed.\n";
                return finish(6);
            }

            // ----------------------------------------------------
            // ADAM
            // ----------------------------------------------------

            if (mem_available_kb() < STEP_MIN_KB) {
                std::cout
                    << "[SAFE STOP] < 1 GiB before Adam.\n";
                return finish(0);
            }

            adam.step(params, grads);

            // ----------------------------------------------------
            // PARAMETER CHECK
            // ----------------------------------------------------

            size_t bf16_params = 0;
            size_t invalid_params = 0;

            for (const auto& p : params) {
                if (!p)
                    continue;

                if (p->dtype() != kBFloat16) {
                    ++invalid_params;
                    continue;
                }

                ++bf16_params;

                if (!finite_bf16(p))
                    ++invalid_params;
            }

            std::cout
                << "BF16 parameters: "
                << bf16_params
                << "\n";

            std::cout
                << "Invalid parameters: "
                << invalid_params
                << "\n";

            if (bf16_params != params.size() ||
                invalid_params != 0) {

                std::cerr
                    << "[FAIL] Parameter validation failed.\n";
                return finish(7);
            }

            // ----------------------------------------------------
            // CLEAR GRADIENTS
            // ----------------------------------------------------

            for (auto& p : model.parameters()) {
                if (p)
                    p->zero_grad();
            }

            // Release graph references before next step.
            loss.reset();
            logits.reset();

            print_mem("after step");
        }

        // --------------------------------------------------------
        // FINAL RESULT
        // --------------------------------------------------------

        std::cout
            << "\n============================================\n"
            << "10-STEP BF16 FULL-FT COMPLETE\n"
            << "============================================\n"
            << "First loss: " << first_loss << "\n"
            << "Last loss:  " << last_loss << "\n"
            << "Delta:      "
            << (last_loss - first_loss)
            << "\n";

        if (!std::isfinite(first_loss) ||
            !std::isfinite(last_loss)) {

            std::cerr
                << "[FAIL] Non-finite training loss.\n";
            return finish(8);
        }

        std::cout
            << "[PASS] 10 forward passes.\n"
            << "[PASS] 10 backward passes.\n"
            << "[PASS] 10 FP32-Adam updates.\n"
            << "[PASS] Gradients cleared every step.\n"
            << "[PASS] Parameters remained BF16.\n";

        print_mem("final");

        return finish(0);

    } catch (const std::exception& e) {
        std::cerr
            << "\n[ERROR] "
            << e.what()
            << "\n";

        return finish(10);
    }
}
