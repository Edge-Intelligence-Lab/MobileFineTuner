#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>

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

    std::string key;
    std::string unit;
    long long value = 0;

    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:")
            return value;
    }

    return -1;
}

static void print_mem(const char* name) {
    const long long kb = mem_available_kb();

    std::cout
        << "[MEM] "
        << name
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

/*
 * Dataset format:
 *
 * {"classic":"Tamil text...","description":"..."}
 *
 * Only "classic" is used.
 *
 * This parser handles the JSON escaping used by the current
 * local dataset, including \n, \r, \t, \" and \\.
 */
static bool extract_json_string(
    const std::string& line,
    const std::string& field,
    std::string& result)
{
    const std::string key = "\"" + field + "\"";

    const size_t key_pos =
        line.find(key);

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

    if (!f) {
        throw std::runtime_error(
            "Cannot open training dataset: " + path);
    }

    std::vector<std::string> records;

    std::string line;

    while (std::getline(f, line)) {

        if (line.empty())
            continue;

        std::string classic;

        if (extract_json_string(
                line,
                "classic",
                classic)) {

            if (!classic.empty())
                records.push_back(classic);
        }
    }

    return records;
}

int main() {
    /*
     * Conservative first real-corpus run.
     *
     * 40 real training records exist locally.
     * We perform 20 optimizer updates, cycling through the records.
     *
     * MAX_SEQ is intentionally 32 for the first real-data run.
     * The previous 124M tests were sequence length 8; increasing
     * sequence length substantially increases activation memory.
     */
    constexpr int MAX_STEPS = 20;
    constexpr int64_t MAX_SEQ = 32;
    constexpr int64_t VOCAB_SIZE = 50257;

    /*
     * Safety limits.
     */
    constexpr long long START_MIN_KB =
        1800LL * 1024LL;

    constexpr long long STEP_MIN_KB =
        1000LL * 1024LL;

    constexpr long long HARD_STOP_KB =
        600LL * 1024LL;

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

    auto finish = [&](int code) {
        stop_watchdog = true;
        watchdog.join();
        return code;
    };

    try {
        const std::string model_dir =
            "/root/gpt2-tamil-124m";

        const std::string train_path =
            "/root/classical-tamil-ppe-smoke/train.jsonl";

        std::cout
            << "============================================\n"
            << " NATIVE 124M BF16 REAL-DATA FINE-TUNING\n"
            << "============================================\n";

        print_mem("start");

        if (mem_available_kb() < START_MIN_KB) {

            std::cout
                << "[SAFE STOP] "
                << "Need >= 1.8 GiB at startup.\n";

            return finish(0);
        }

        // ========================================================
        // REAL DATASET
        // ========================================================

        const auto records =
            load_classic_records(train_path);

        std::cout
            << "Training records: "
            << records.size()
            << "\n";

        if (records.empty())
            throw std::runtime_error(
                "Training dataset contains no classic records.");

        // ========================================================
        // NATIVE GPT-2 TOKENIZER
        // ========================================================

        TokenizerLoadOptions tokenizer_options;

        tokenizer_options.model_type = "gpt2";

        auto tokenizer =
            TokenizerFactory::from_pretrained(
                model_dir,
                tokenizer_options);

        if (!tokenizer)
            throw std::runtime_error(
                "Native GPT-2 tokenizer failed to load.");

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
        // BF16 MODEL LOAD
        // ========================================================

        SafeTensorsModelReader reader(
            model_dir);

        reader.parse_headers();

        auto mapping =
            GPT2KeyMapper::generate_gpt2_mapping(
                config.n_layer);

        SafeTensorsLoadOptions load_options;

        load_options.transpose_linear = false;
        load_options.auto_promote_fp16 = true;
        load_options.convert_f32_to_bf16 = true;
        load_options.verbose = false;

        auto tensors =
            reader.load_tensors_mapped(
                mapping,
                load_options);

        size_t total_params = 0;
        size_t bf16_tensors = 0;
        size_t fp32_tensors = 0;

        for (const auto& kv : tensors) {

            if (!kv.second)
                continue;

            total_params +=
                static_cast<size_t>(
                    kv.second->numel());

            if (kv.second->dtype() == kBFloat16)
                ++bf16_tensors;

            else if (kv.second->dtype() == kFloat32)
                ++fp32_tensors;

            model.assign_weight(
                kv.first,
                kv.second);
        }

        std::cout
            << "Parameters:    "
            << total_params
            << "\n";

        std::cout
            << "BF16 tensors:  "
            << bf16_tensors
            << "\n";

        std::cout
            << "FP32 tensors:  "
            << fp32_tensors
            << "\n";

        if (total_params != 124439808 ||
            bf16_tensors != 148 ||
            fp32_tensors != 0) {

            std::cerr
                << "[FAIL] Expected 124M BF16 model.\n";

            return finish(2);
        }

        std::cout
            << "[PASS] 124M parameters are BF16.\n";

        print_mem("after model load");

        // ========================================================
        // FULL PARAMETER TRAINING
        // ========================================================

        for (auto& p :
             model.parameters()) {

            if (p)
                p->set_requires_grad(true);
        }

        // ========================================================
        // ADAM
        // ========================================================

        AdamConfig adam_config;

        adam_config.learning_rate = 5e-6f;
        adam_config.beta1 = 0.9f;
        adam_config.beta2 = 0.999f;
        adam_config.epsilon = 1e-8f;
        adam_config.weight_decay = 0.0f;
        adam_config.amsgrad = false;

        Adam adam(adam_config);

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        bool have_loss = false;

        int completed_steps = 0;

        // ========================================================
        // REAL-DATA CAUSAL-LM LOOP
        // ========================================================

        for (int step = 0;
             step < MAX_STEPS;
             ++step) {

            std::cout
                << "\n--------------------------------------------\n"
                << "STEP "
                << (step + 1)
                << "/"
                << MAX_STEPS
                << "\n"
                << "--------------------------------------------\n";

            print_mem("before step");

            if (mem_available_kb() < STEP_MIN_KB) {

                std::cout
                    << "[SAFE STOP] "
                    << "Memory below 1 GiB.\n";

                break;
            }

            const std::string& text =
                records[
                    static_cast<size_t>(step)
                    % records.size()
                ];

            // ====================================================
            // TOKENIZE REAL TAMIL TEXT
            // ====================================================

            const std::vector<int> ids =
                tokenizer->encode(text);

            if (ids.size() < 2) {

                std::cout
                    << "[SKIP] Record produced <2 tokens.\n";

                continue;
            }

            /*
             * We need N input tokens and N target tokens:
             *
             * input : t0 t1 t2 ... t(N-1)
             * target: t1 t2 t3 ... tN
             *
             * Therefore usable token count is MAX_SEQ.
             */
            const size_t usable =
                std::min(
                    ids.size(),
                    static_cast<size_t>(
                        MAX_SEQ + 1));

            if (usable < 2) {

                std::cout
                    << "[SKIP] Record too short.\n";

                continue;
            }

            const int64_t seq_len =
                static_cast<int64_t>(
                    usable - 1);

            std::vector<int64_t>
                input_values(
                    static_cast<size_t>(
                        seq_len));

            std::vector<int32_t>
                label_values(
                    static_cast<size_t>(
                        seq_len));

            for (int64_t i = 0;
                 i < seq_len;
                 ++i) {

                input_values[
                    static_cast<size_t>(i)]
                    = static_cast<int64_t>(
                        ids[
                            static_cast<size_t>(i)]);

                label_values[
                    static_cast<size_t>(i)]
                    = static_cast<int32_t>(
                        ids[
                            static_cast<size_t>(i + 1)]);
            }

            // ====================================================
            // INPUT / LABEL TENSORS
            // ====================================================

            auto input_ids =
                std::make_shared<Tensor>(
                    std::vector<int64_t>{
                        1,
                        seq_len
                    },
                    input_values.data(),
                    kInt64,
                    kCPU);

            auto labels =
                std::make_shared<Tensor>(
                    std::vector<int64_t>{
                        1,
                        seq_len
                    },
                    label_values.data(),
                    kInt32,
                    kCPU);

            // ====================================================
            // NATIVE GPT-2 FORWARD
            // ====================================================

            auto logits =
                model.forward(input_ids);

            if (!logits ||
                logits->dtype() != kFloat32 ||
                logits->shape().size() != 3 ||
                logits->shape()[0] != 1 ||
                logits->shape()[1] != seq_len ||
                logits->shape()[2] != VOCAB_SIZE ||
                !finite_f32(logits)) {

                std::cerr
                    << "[FAIL] Invalid logits at step "
                    << (step + 1)
                    << "\n";

                return finish(4);
            }

            // ====================================================
            // REAL CAUSAL LANGUAGE-MODELING OBJECTIVE
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
                    << (step + 1)
                    << "\n";

                return finish(5);
            }

            const float loss_value =
                loss->item();

            if (!have_loss) {
                first_loss = loss_value;
                have_loss = true;
            }

            last_loss = loss_value;

            std::cout
                << "Tokens: "
                << seq_len
                << "\n";

            std::cout
                << "Causal-LM loss: "
                << loss_value
                << "\n";

            // ====================================================
            // BACKWARD
            // ====================================================

            if (mem_available_kb() < STEP_MIN_KB) {

                std::cout
                    << "[SAFE STOP] "
                    << "Memory below 1 GiB before backward.\n";

                break;
            }

            loss->backward();

            std::vector<TensorPtr>
                params;

            std::vector<TensorPtr>
                grads;

            params.reserve(
                model.parameters().size());

            grads.reserve(
                model.parameters().size());

            size_t grad_count = 0;
            size_t invalid_grads = 0;

            for (const auto& p :
                 model.parameters()) {

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
                    << "[FAIL] Invalid gradient set.\n";

                return finish(6);
            }

            // ====================================================
            // FP32 ADAM → BF16 PARAMETERS
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
            // VERIFY BF16 PARAMETERS
            // ====================================================

            size_t bf16_params = 0;
            size_t invalid_params = 0;

            for (const auto& p :
                 params) {

                if (!p ||
                    p->dtype() != kBFloat16) {

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
                    << "[FAIL] BF16 parameter verification failed.\n";

                return finish(7);
            }

            // ====================================================
            // CLEAR GRADIENTS
            // ====================================================

            for (auto& p :
                 model.parameters()) {

                if (p)
                    p->zero_grad();
            }

            /*
             * Release this step's graph before the next dataset
             * record. This is essential for stable long-running
             * training on the phone.
             */
            loss.reset();
            logits.reset();
            input_ids.reset();
            labels.reset();

            ++completed_steps;

            print_mem("after step");
        }

        // ========================================================
        // RESULT
        // ========================================================

        std::cout
            << "\n============================================\n"
            << "NATIVE REAL-DATA BF16 TRAINING RESULT\n"
            << "============================================\n";

        std::cout
            << "Completed steps: "
            << completed_steps
            << "\n";

        if (have_loss) {

            std::cout
                << "First causal-LM loss: "
                << first_loss
                << "\n";

            std::cout
                << "Last causal-LM loss:  "
                << last_loss
                << "\n";

            std::cout
                << "Loss delta: "
                << (last_loss - first_loss)
                << "\n";
        }

        if (completed_steps == 0) {

            std::cerr
                << "[FAIL] No training steps completed.\n";

            return finish(8);
        }

        if (!have_loss ||
            !std::isfinite(first_loss) ||
            !std::isfinite(last_loss)) {

            std::cerr
                << "[FAIL] Invalid final loss.\n";

            return finish(9);
        }

        std::cout
            << "[PASS] Real JSONL Classical-Tamil data.\n"
            << "[PASS] Native GPT-2 tokenizer.\n"
            << "[PASS] Causal next-token targets.\n"
            << "[PASS] Native lm_cross_entropy.\n"
            << "[PASS] Native backward.\n"
            << "[PASS] FP32 gradients.\n"
            << "[PASS] FP32 Adam.\n"
            << "[PASS] BF16 parameter write-back.\n"
            << "[PASS] Gradient reset every step.\n";

        // ========================================================
        // SAVE TRAINED BF16 MODEL
        // ========================================================
        //
        // Native checkpoint format:
        //
        //   magic[8]
        //   tensor_count(uint64)
        //
        //   repeated:
        //     key_length(uint64)
        //     key bytes
        //     dtype(uint32)
        //     ndim(uint64)
        //     shape[ndim](int64)
        //     raw tensor bytes(uint64 + data)
        //
        // No FP32 conversion is performed.
        //
        const std::string output_checkpoint =
            "/root/tamil_gpt2_bf16_trained.bin";

        std::cout
            << "\n===== SAVING TRAINED BF16 MODEL =====\n";

        {
            std::ofstream out(
                output_checkpoint,
                std::ios::binary);

            if (!out)
                throw std::runtime_error(
                    "Cannot create trained BF16 checkpoint.");

            const char magic[8] = {
                'M','F','T','B','F','1','6','1'
            };

            out.write(
                magic,
                sizeof(magic));

            uint64_t tensor_count = 0;

            for (const auto& kv : tensors) {
                if (kv.second)
                    ++tensor_count;
            }

            out.write(
                reinterpret_cast<const char*>(&tensor_count),
                sizeof(tensor_count));

            for (const auto& kv : tensors) {
                const std::string& key = kv.first;
                const TensorPtr& t = kv.second;

                if (!t)
                    continue;

                if (t->dtype() != kBFloat16) {
                    throw std::runtime_error(
                        "Attempted to save non-BF16 tensor: " + key);
                }

                const uint64_t key_len =
                    static_cast<uint64_t>(key.size());

                out.write(
                    reinterpret_cast<const char*>(&key_len),
                    sizeof(key_len));

                out.write(
                    key.data(),
                    static_cast<std::streamsize>(key.size()));

                const uint32_t dtype =
                    static_cast<uint32_t>(t->dtype());

                out.write(
                    reinterpret_cast<const char*>(&dtype),
                    sizeof(dtype));

                const uint64_t ndim =
                    static_cast<uint64_t>(
                        t->shape().size());

                out.write(
                    reinterpret_cast<const char*>(&ndim),
                    sizeof(ndim));

                for (const auto dim : t->shape()) {
                    const int64_t d =
                        static_cast<int64_t>(dim);

                    out.write(
                        reinterpret_cast<const char*>(&d),
                        sizeof(d));
                }

                const uint64_t data_bytes =
                    static_cast<uint64_t>(
                        t->numel() * sizeof(uint16_t));

                out.write(
                    reinterpret_cast<const char*>(&data_bytes),
                    sizeof(data_bytes));

                out.write(
                    reinterpret_cast<const char*>(
                        t->data<uint16_t>()),
                    static_cast<std::streamsize>(
                        data_bytes));

                if (!out)
                    throw std::runtime_error(
                        "Write failed for tensor: " + key);
            }

            out.flush();

            if (!out)
                throw std::runtime_error(
                    "Failed flushing trained checkpoint.");
        }

        // Verify the file exists and is non-empty.
        {
            std::ifstream in(
                output_checkpoint,
                std::ios::binary | std::ios::ate);

            if (!in)
                throw std::runtime_error(
                    "Cannot reopen saved checkpoint.");

            const auto size = in.tellg();

            if (size <= 0)
                throw std::runtime_error(
                    "Saved checkpoint is empty.");

            std::cout
                << "Saved checkpoint: "
                << output_checkpoint
                << "\n";

            std::cout
                << "Checkpoint bytes: "
                << size
                << "\n";
        }

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
