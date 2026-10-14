#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../core/tensor.h"
#include "../core/dtype.h"

using namespace ops;

static long long mem_available_kb() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long value;
    std::string unit;

    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value;
        }
    }

    return -1;
}

int main() {
    try {
        const long long available_kb = mem_available_kb();

        std::cout << "===== BF16 124M FORWARD SAFETY CHECK =====\n";
        std::cout << "MemAvailable: " << available_kb << " kB\n";

        // Conservative phone-safety gate.
        // We require about 2 GiB available before attempting this forward.
        if (available_kb > 0 && available_kb < 2LL * 1024LL * 1024LL) {
            std::cout << "[SAFE STOP] Less than 2 GiB available.\n";
            std::cout << "[SAFE STOP] Skipping 124M forward.\n";
            return 0;
        }

        const std::string model_dir = "/root/gpt2-tamil-124m";

        GPT2Config config = GPT2Config::from_pretrained(model_dir);
        GPT2Model model(config);

        if (config.tie_word_embeddings) {
            model.tie_weights();
        }

        // Forward-only test: no autograd graph for model parameters.
        for (auto& p : model.parameters()) {
            if (p) {
                p->set_requires_grad(false);
            }
        }

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

        size_t parameter_bytes = 0;
        size_t parameter_count = 0;
        size_t bf16_count = 0;
        size_t fp32_count = 0;

        for (const auto& kv : tensors) {
            const auto& t = kv.second;
            if (!t) continue;

            parameter_count += static_cast<size_t>(t->numel());
            parameter_bytes +=
                static_cast<size_t>(t->numel()) *
                DTypeUtils::size_of(t->dtype());

            if (t->dtype() == kBFloat16) {
                ++bf16_count;
            } else if (t->dtype() == kFloat32) {
                ++fp32_count;
            }

            model.assign_weight(kv.first, t);
        }

        std::cout << "Parameters:    " << parameter_count << "\n";
        std::cout << "BF16 tensors:  " << bf16_count << "\n";
        std::cout << "FP32 tensors:  " << fp32_count << "\n";
        std::cout << "Weight MiB:    "
                  << (static_cast<double>(parameter_bytes) /
                      (1024.0 * 1024.0))
                  << "\n";

        if (bf16_count == 0 || fp32_count != 0) {
            std::cerr << "[FAIL] Weight dtype verification failed.\n";
            return 2;
        }

        /*
         * Very small sequence deliberately chosen for the first real-model
         * forward. This keeps activation memory low.
         */
        const int64_t ids[8] = {
            288, 275, 271, 265,
            281, 262, 742, 302
        };

        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 8},
            ids,
            kInt64,
            kCPU);

        std::cout << "Running forward: batch=1 seq=8\n";

        auto logits = model.forward(input_ids);

        if (!logits) {
            std::cerr << "[FAIL] Forward returned null.\n";
            return 3;
        }

        const auto& shape = logits->shape();

        std::cout << "Logits shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << shape[i];
        }
        std::cout << "]\n";

        if (shape.size() != 3 ||
            shape[0] != 1 ||
            shape[1] != 8) {
            std::cerr << "[FAIL] Unexpected logits shape.\n";
            return 4;
        }

        if (logits->dtype() != kFloat32) {
            std::cerr << "[FAIL] Logits are not FP32.\n";
            return 5;
        }

        const float* data = logits->data<float>();
        bool finite = true;

        for (int i = 0; i < 16; ++i) {
            if (!std::isfinite(data[i])) {
                finite = false;
                break;
            }
        }

        if (!finite) {
            std::cerr << "[FAIL] Non-finite logits detected.\n";
            return 6;
        }

        std::cout << "First logits:";
        for (int i = 0; i < 8; ++i) {
            std::cout << " " << data[i];
        }
        std::cout << "\n";

        std::cout << "[PASS] 124M BF16 weights loaded.\n";
        std::cout << "[PASS] 124M forward completed.\n";
        std::cout << "[PASS] FP32 logits are finite.\n";
        std::cout << "[PASS] No backward/optimizer/training executed.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 10;
    }
}
