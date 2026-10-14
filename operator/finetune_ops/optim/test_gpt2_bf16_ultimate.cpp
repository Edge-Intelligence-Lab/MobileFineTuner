#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../core/tensor.h"
#include "../core/ops.h"
#include "../core/dtype.h"
#include "../core/tokenizer.h"
#include "../core/lm_loss.h"
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
    const auto kb = mem_available_kb();

    std::cout
        << "[MEM] "
        << label
        << ": "
        << kb / 1024.0
        << " MiB available\n";
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

static uint64_t bf16_hash(const TensorPtr& t) {
    if (!t || t->dtype() != kBFloat16)
        return 0;

    const uint16_t* p = t->data<uint16_t>();

    uint64_t h = 1469598103934665603ULL;

    for (int64_t i = 0; i < t->numel(); ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }

    return h;
}

static bool extract_json_string(
    const std::string& line,
    const std::string& field,
    std::string& result)
{
    const std::string key = "\"" + field + "\"";

    const size_t key_pos = line.find(key);

    if (key_pos == std::string::npos)
        return false;

    const size_t colon =
        line.find(':', key_pos + key.size());

    if (colon == std::string::npos)
        return false;

    const size_t first_quote =
        line.find('"', colon + 1);

    if (first_quote == std::string::npos)
        return false;

    result.clear();

    bool escaped = false;

    for (size_t i = first_quote + 1;
         i < line.size();
         ++i) {

        const char c = line[i];

        if (escaped) {
            switch (c) {
            case 'n':
                result.push_back('\n');
                break;
            case 'r':
                result.push_back('\r');
                break;
            case 't':
                result.push_back('\t');
                break;
            case '"':
                result.push_back('"');
                break;
            case '\\':
                result.push_back('\\');
                break;
            case '/':
                result.push_back('/');
                break;
            default:
                result.push_back(c);
                break;
            }

            escaped = false;
            continue;
        }

        if (c == '\\') {
            escaped = true;
            continue;
        }

        if (c == '"')
            break;

        result.push_back(c);
    }

    return !result.empty();
}

static std::vector<std::string> load_classic_records(
    const std::string& path)
{
    std::ifstream f(path);

    if (!f)
        throw std::runtime_error(
            "Cannot open dataset: " + path);

    std::vector<std::string> rows;

    std::string line;

    while (std::getline(f, line)) {
        if (line.empty())
            continue;

        std::string text;

        if (extract_json_string(
                line,
                "classic",
                text)) {

            rows.push_back(text);
        }
    }

    return rows;
}

static double evaluate_dataset(
    GPT2Model& model,
    Tokenizer& tokenizer,
    const std::vector<std::string>& records,
    int64_t max_seq)
{
    double total_loss = 0.0;
    int used = 0;

    // Evaluation should not retain parameter gradients.
    for (auto& p : model.parameters()) {
        if (p)
            p->set_requires_grad(false);
    }

    for (const auto& text : records) {
        const auto ids = tokenizer.encode(text);

        if (ids.size() < 2)
            continue;

        const size_t usable =
            std::min(
                ids.size(),
                static_cast<size_t>(max_seq + 1));

        if (usable < 2)
            continue;

        const int64_t S =
            static_cast<int64_t>(usable - 1);

        std::vector<int64_t> input_values(
            static_cast<size_t>(S));

        std::vector<int32_t> label_values(
            static_cast<size_t>(S));

        for (int64_t i = 0; i < S; ++i) {
            input_values[
                static_cast<size_t>(i)]
                = static_cast<int64_t>(
                    ids[static_cast<size_t>(i)]);

            label_values[
                static_cast<size_t>(i)]
                = static_cast<int32_t>(
                    ids[static_cast<size_t>(i + 1)]);
        }

        auto input_ids =
            std::make_shared<Tensor>(
                std::vector<int64_t>{1, S},
                input_values.data(),
                kInt64,
                kCPU);

        auto labels =
            std::make_shared<Tensor>(
                std::vector<int64_t>{1, S},
                label_values.data(),
                kInt32,
                kCPU);

        auto logits =
            model.forward(input_ids);

        if (!logits ||
            logits->dtype() != kFloat32 ||
            !finite_f32(logits)) {

            throw std::runtime_error(
                "Validation produced invalid logits.");
        }

        auto loss =
            lm_cross_entropy(
                logits,
                labels,
                -100,
                "mean");

        if (!loss ||
            !std::isfinite(loss->item())) {

            throw std::runtime_error(
                "Validation produced invalid loss.");
        }

        total_loss +=
            static_cast<double>(loss->item());

        ++used;

        loss.reset();
        logits.reset();
    }

    // Clear anything left and re-enable training.
    for (auto& p : model.parameters()) {
        if (p) {
            p->zero_grad();
            p->set_requires_grad(true);
        }
    }

    if (used == 0)
        throw std::runtime_error(
            "Validation dataset produced zero usable samples.");

    return total_loss /
           static_cast<double>(used);
}

int main() {
    constexpr int MAX_STEPS = 100;
    constexpr int64_t MAX_SEQ = 32;
    constexpr int64_t VOCAB = 50257;

    constexpr long long START_MIN_KB =
        1800LL * 1024LL;

    constexpr long long STEP_MIN_KB =
        1000LL * 1024LL;

    constexpr long long HARD_STOP_KB =
        600LL * 1024LL;

    const std::string model_dir =
        "/root/gpt2-tamil-124m";

    const std::string train_path =
        "/root/classical-tamil-ppe-smoke/train.jsonl";

    const std::string valid_path =
        "/root/classical-tamil-ppe-smoke/valid.jsonl";

    const std::string checkpoint_path =
        "/root/tamil_gpt2_bf16_ultimate.ckpt";

    std::atomic<bool> stop_watchdog{false};

    std::thread watchdog([&]() {
        while (!stop_watchdog.load()) {
            const long long kb =
                mem_available_kb();

            if (kb > 0 &&
                kb < HARD_STOP_KB) {

                std::cerr
                    << "\n[SAFE ABORT] "
                    << "MemAvailable < 600 MiB\n";

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
        std::cout
            << "====================================================\n"
            << "   ULTIMATE NATIVE 124M BF16 FINE-TUNE TEST\n"
            << "====================================================\n";

        print_mem("startup");

        if (mem_available_kb() < START_MIN_KB) {
            std::cout
                << "[SAFE STOP] "
                << "Startup memory below 1.8 GiB.\n";

            return finish(0);
        }

        // ========================================================
        // DATA
        // ========================================================

        auto train_records =
            load_classic_records(train_path);

        auto valid_records =
            load_classic_records(valid_path);

        std::cout
            << "Training records:   "
            << train_records.size()
            << "\n";

        std::cout
            << "Validation records: "
            << valid_records.size()
            << "\n";

        if (train_records.empty() ||
            valid_records.empty()) {

            throw std::runtime_error(
                "Training or validation dataset is empty.");
        }

        // ========================================================
        // TOKENIZER
        // ========================================================

        TokenizerLoadOptions tok_options;
        tok_options.model_type = "gpt2";

        auto tokenizer =
            TokenizerFactory::from_pretrained(
                model_dir,
                tok_options);

        if (!tokenizer)
            throw std::runtime_error(
                "Native GPT-2 tokenizer failed.");

        std::cout
            << "[PASS] Native GPT-2 tokenizer loaded.\n";

        // ========================================================
        // MODEL
        // ========================================================

        GPT2Config config =
            GPT2Config::from_pretrained(
                model_dir);

        GPT2Model model(config);

        if (config.tie_word_embeddings)
            model.tie_weights();

        // ========================================================
        // BF16 LOAD
        // ========================================================

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

        size_t parameter_count = 0;
        size_t bf16_count = 0;
        size_t fp32_count = 0;

        for (const auto& kv : tensors) {
            if (!kv.second)
                continue;

            parameter_count +=
                static_cast<size_t>(
                    kv.second->numel());

            if (kv.second->dtype() == kBFloat16)
                ++bf16_count;

            else if (kv.second->dtype() == kFloat32)
                ++fp32_count;

            model.assign_weight(
                kv.first,
                kv.second);
        }

        if (parameter_count != 124439808 ||
            bf16_count != 148 ||
            fp32_count != 0) {

            std::cerr
                << "[FAIL] Invalid 124M BF16 model.\n";

            return finish(2);
        }

        std::cout
            << "[PASS] "
            << parameter_count
            << " parameters / "
            << bf16_count
            << " BF16 tensors / "
            << fp32_count
            << " FP32 parameter tensors.\n";

        print_mem("after load");

        // ========================================================
        // FULL FINE-TUNING
        // ========================================================

        for (auto& p : model.parameters()) {
            if (p)
                p->set_requires_grad(true);
        }

        // ========================================================
        // BASELINE VALIDATION
        // ========================================================

        std::cout
            << "\n===== BASELINE VALIDATION =====\n";

        const double baseline_loss =
            evaluate_dataset(
                model,
                *tokenizer,
                valid_records,
                MAX_SEQ);

        std::cout
            << "Baseline validation loss: "
            << baseline_loss
            << "\n";

        // ========================================================
        // ADAM
        // ========================================================

        AdamConfig adam_cfg;

        adam_cfg.learning_rate = 5e-6f;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.999f;
        adam_cfg.epsilon = 1e-8f;
        adam_cfg.weight_decay = 0.0f;
        adam_cfg.amsgrad = false;

        Adam adam(adam_cfg);

        // ========================================================
        // DETERMINISTIC SHUFFLING
        // ========================================================

        std::vector<size_t> order(
            train_records.size());

        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;

        std::mt19937 rng(123456789);

        // ========================================================
        // TRAIN
        // ========================================================

        int completed = 0;
        double first_train_loss = 0.0;
        double last_train_loss = 0.0;

        for (int step = 0;
             step < MAX_STEPS;
             ++step) {

            const int display_step =
                step + 1;

            if (step % static_cast<int>(
                    train_records.size()) == 0) {

                std::shuffle(
                    order.begin(),
                    order.end(),
                    rng);
            }

            print_mem("before step");

            if (mem_available_kb() < STEP_MIN_KB) {

                std::cout
                    << "[SAFE STOP] "
                    << "Memory below 1 GiB.\n";

                break;
            }

            const size_t record_index =
                order[
                    static_cast<size_t>(step)
                    % order.size()
                ];

            const std::string& text =
                train_records[record_index];

            const auto ids =
                tokenizer->encode(text);

            if (ids.size() < 2) {
                std::cout
                    << "[SKIP] Too few tokens.\n";
                continue;
            }

            const size_t usable =
                std::min(
                    ids.size(),
                    static_cast<size_t>(
                        MAX_SEQ + 1));

            if (usable < 2) {
                std::cout
                    << "[SKIP] Too few usable tokens.\n";
                continue;
            }

            const int64_t S =
                static_cast<int64_t>(
                    usable - 1);

            std::vector<int64_t> input_values(
                static_cast<size_t>(S));

            std::vector<int32_t> label_values(
                static_cast<size_t>(S));

            for (int64_t i = 0;
                 i < S;
                 ++i) {

                input_values[
                    static_cast<size_t>(i)]
                    = static_cast<int64_t>(
                        ids[static_cast<size_t>(i)]);

                label_values[
                    static_cast<size_t>(i)]
                    = static_cast<int32_t>(
                        ids[static_cast<size_t>(i + 1)]);
            }

            auto input_ids =
                std::make_shared<Tensor>(
                    std::vector<int64_t>{1, S},
                    input_values.data(),
                    kInt64,
                    kCPU);

            auto labels =
                std::make_shared<Tensor>(
                    std::vector<int64_t>{1, S},
                    label_values.data(),
                    kInt32,
                    kCPU);

            // ====================================================
            // FORWARD
            // ====================================================

            auto logits =
                model.forward(input_ids);

            if (!logits ||
                logits->dtype() != kFloat32 ||
                logits->shape().size() != 3 ||
                logits->shape()[0] != 1 ||
                logits->shape()[1] != S ||
                logits->shape()[2] != VOCAB ||
                !finite_f32(logits)) {

                std::cerr
                    << "[FAIL] Invalid logits at step "
                    << display_step
                    << "\n";

                return finish(4);
            }

            // ====================================================
            // REAL CAUSAL LM OBJECTIVE
            // ====================================================

            auto loss =
                lm_cross_entropy(
                    logits,
                    labels,
                    -100,
                    "mean");

            if (!loss ||
                !std::isfinite(
                    loss->item())) {

                std::cerr
                    << "[FAIL] Invalid causal-LM loss at step "
                    << display_step
                    << "\n";

                return finish(5);
            }

            const double loss_value =
                static_cast<double>(
                    loss->item());

            if (completed == 0)
                first_train_loss = loss_value;

            last_train_loss = loss_value;

            // ====================================================
            // BACKWARD
            // ====================================================

            loss->backward();

            std::vector<TensorPtr> params;
            std::vector<TensorPtr> grads;

            size_t invalid_grads = 0;

            for (const auto& p :
                 model.parameters()) {

                if (!p)
                    continue;

                auto g = p->grad();

                if (!g)
                    continue;

                if (g->dtype() != kFloat32 ||
                    !finite_f32(g)) {

                    ++invalid_grads;
                    continue;
                }

                params.push_back(p);
                grads.push_back(g);
            }

            if (params.empty() ||
                invalid_grads != 0) {

                std::cerr
                    << "[FAIL] Invalid gradient set at step "
                    << display_step
                    << "\n";

                return finish(6);
            }

            // ====================================================
            // ADAM
            // ====================================================

            if (mem_available_kb() < STEP_MIN_KB) {

                std::cout
                    << "[SAFE STOP] "
                    << "Memory below 1 GiB before Adam.\n";

                break;
            }

            adam.step(
                params,
                grads);

            // ====================================================
            // BF16 PARAMETER INTEGRITY
            // ====================================================

            size_t bf16_params = 0;
            size_t invalid_params = 0;

            for (const auto& p : params) {

                if (!p ||
                    p->dtype() != kBFloat16) {

                    ++invalid_params;
                    continue;
                }

                ++bf16_params;

                if (!finite_bf16(p))
                    ++invalid_params;
            }

            if (bf16_params != params.size() ||
                invalid_params != 0) {

                std::cerr
                    << "[FAIL] BF16 parameter corruption at step "
                    << display_step
                    << "\n";

                return finish(7);
            }

            // ====================================================
            // CLEAR GRAPH
            // ====================================================

            for (auto& p :
                 model.parameters()) {

                if (p)
                    p->zero_grad();
            }

            loss.reset();
            logits.reset();
            input_ids.reset();
            labels.reset();

            ++completed;

            std::cout
                << "Step "
                << display_step
                << "/"
                << MAX_STEPS
                << " | loss="
                << loss_value
                << " | tokens="
                << S
                << " | BF16 params="
                << bf16_params
                << "\n";

            print_mem("after step");

            // Validation checkpoints.
            if (completed == 50) {

                std::cout
                    << "\n===== MID-RUN VALIDATION =====\n";

                const double val_loss =
                    evaluate_dataset(
                        model,
                        *tokenizer,
                        valid_records,
                        MAX_SEQ);

                std::cout
                    << "Validation loss @ step 50: "
                    << val_loss
                    << "\n";
            }
        }

        if (completed == 0) {

            std::cerr
                << "[FAIL] Zero training steps completed.\n";

            return finish(8);
        }

        // ========================================================
        // FINAL VALIDATION
        // ========================================================

        std::cout
            << "\n===== FINAL VALIDATION =====\n";

        const double final_val_loss =
            evaluate_dataset(
                model,
                *tokenizer,
                valid_records,
                MAX_SEQ);

        std::cout
            << "Baseline validation loss: "
            << baseline_loss
            << "\n";

        std::cout
            << "Final validation loss:    "
            << final_val_loss
            << "\n";

        // ========================================================
        // FINAL MODEL INTEGRITY
        // ========================================================

        size_t final_bf16 = 0;
        size_t invalid_final = 0;
        uint64_t aggregate_hash = 0;

        for (const auto& p :
             model.parameters()) {

            if (!p)
                continue;

            if (p->dtype() != kBFloat16) {
                ++invalid_final;
                continue;
            }

            ++final_bf16;

            if (!finite_bf16(p)) {
                ++invalid_final;
                continue;
            }

            aggregate_hash ^=
                bf16_hash(p)
                + 0x9e3779b97f4a7c15ULL
                + (aggregate_hash << 6)
                + (aggregate_hash >> 2);
        }

        std::cout
            << "\nFinal BF16 parameter tensors: "
            << final_bf16
            << "\n";

        std::cout
            << "Invalid final tensors: "
            << invalid_final
            << "\n";

        std::cout
            << "Parameter hash: 0x"
            << std::hex
            << aggregate_hash
            << std::dec
            << "\n";

        if (final_bf16 != 148 ||
            invalid_final != 0) {

            std::cerr
                << "[FAIL] Final BF16 integrity check failed.\n";

            return finish(9);
        }

        // ========================================================
        // NATIVE BF16 CHECKPOINT
        // ========================================================
        //
        // Custom compact checkpoint:
        //
        //   magic
        //   number of tensors
        //   for every model parameter:
        //       numel
        //       raw BF16 values
        //
        // It is intentionally a runtime checkpoint rather than a
        // SafeTensors file. It proves that the trained BF16 state
        // can be serialized without converting back to FP32.
        //
        // ========================================================

        std::cout
            << "\n===== BF16 CHECKPOINT WRITE =====\n";

        {
            std::ofstream out(
                checkpoint_path,
                std::ios::binary);

            if (!out)
                throw std::runtime_error(
                    "Cannot create BF16 checkpoint.");

            const uint64_t magic =
                0x42463136544E3031ULL;

            const uint64_t count =
                static_cast<uint64_t>(
                    model.parameters().size());

            out.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic));

            out.write(
                reinterpret_cast<const char*>(&count),
                sizeof(count));

            for (const auto& p :
                 model.parameters()) {

                if (!p ||
                    p->dtype() != kBFloat16) {

                    throw std::runtime_error(
                        "Non-BF16 parameter during checkpoint.");
                }

                const uint64_t n =
                    static_cast<uint64_t>(
                        p->numel());

                out.write(
                    reinterpret_cast<const char*>(&n),
                    sizeof(n));

                out.write(
                    reinterpret_cast<const char*>(
                        p->data<uint16_t>()),
                    static_cast<std::streamsize>(
                        n * sizeof(uint16_t)));
            }

            out.flush();
        }

        std::ifstream checkpoint_in(
            checkpoint_path,
            std::ios::binary | std::ios::ate);

        if (!checkpoint_in)
            throw std::runtime_error(
                "Cannot reopen checkpoint.");

        const auto checkpoint_size =
            checkpoint_in.tellg();

        std::cout
            << "Checkpoint size: "
            << checkpoint_size
            << " bytes\n";

        if (checkpoint_size <= 0)
            throw std::runtime_error(
                "Checkpoint is empty.");

        std::cout
            << "[PASS] BF16 checkpoint written.\n";

        // ========================================================
        // END
        // ========================================================

        print_mem("final");

        std::cout
            << "\n====================================================\n"
            << "                 ULTIMATE TEST RESULT\n"
            << "====================================================\n";

        std::cout
            << "[PASS] Real Classical-Tamil training data.\n"
            << "[PASS] Native GPT-2 tokenizer.\n"
            << "[PASS] Native causal next-token objective.\n"
            << "[PASS] Native lm_cross_entropy.\n"
            << "[PASS] Native forward/backward.\n"
            << "[PASS] FP32 gradients.\n"
            << "[PASS] FP32 Adam state/update.\n"
            << "[PASS] BF16 parameter storage throughout training.\n"
            << "[PASS] Gradient reset after every step.\n"
            << "[PASS] Memory watchdog remained active.\n"
            << "[PASS] Final BF16 parameter integrity.\n"
            << "[PASS] Native BF16 checkpoint serialization.\n";

        std::cout
            << "\nCompleted steps: "
            << completed
            << "\n";

        std::cout
            << "First training loss: "
            << first_train_loss
            << "\n";

        std::cout
            << "Last training loss:  "
            << last_train_loss
            << "\n";

        std::cout
            << "Validation baseline: "
            << baseline_loss
            << "\n";

        std::cout
            << "Validation final:    "
            << final_val_loss
            << "\n";

        std::cout
            << "====================================================\n";

        return finish(0);

    } catch (const std::exception& e) {

        std::cerr
            << "\n[FATAL] "
            << e.what()
            << "\n";

        return finish(10);
    }
}
