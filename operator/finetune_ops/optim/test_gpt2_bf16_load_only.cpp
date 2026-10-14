#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../graph/safetensors_loader.h"
#include "../graph/gpt2_model.h"

using namespace ops;

int main() {
    const std::string model_dir = "/root/gpt2-tamil-124m";

    try {
        GPT2Config config = GPT2Config::from_pretrained(model_dir);
        GPT2Model model(config);

        if (config.tie_word_embeddings) {
            model.tie_weights();
        }

        SafeTensorsModelReader reader(model_dir);
        reader.parse_headers();

        auto mapping = GPT2KeyMapper::generate_gpt2_mapping(config.n_layer);

        SafeTensorsLoadOptions options;
        options.transpose_linear = false;
        options.auto_promote_fp16 = true;
        options.convert_f32_to_bf16 = true;
        options.verbose = false;

        /*
         * Critical safety feature:
         * request low-precision storage for every mapped GPT-2 weight.
         *
         * The loader must support F32 -> BF16 conversion for this to pass.
         */
        options.preserve_low_precision_key_substrings.clear();

        auto tensors = reader.load_tensors_mapped(mapping, options);

        size_t total_params = 0;
        size_t total_bytes = 0;
        size_t bf16_tensors = 0;
        size_t fp32_tensors = 0;

        for (const auto& kv : tensors) {
            const auto& name = kv.first;
            const auto& tensor = kv.second;

            total_params += static_cast<size_t>(tensor->numel());
            total_bytes += static_cast<size_t>(tensor->numel()) *
                           DTypeUtils::size_of(tensor->dtype());

            if (tensor->dtype() == kBFloat16) {
                ++bf16_tensors;
            } else if (tensor->dtype() == kFloat32) {
                ++fp32_tensors;
            }

            std::cout << "[BF16-LOAD] " << name
                      << " dtype=" << DTypeUtils::to_string(tensor->dtype())
                      << " numel=" << tensor->numel()
                      << " bytes="
                      << (tensor->numel() * DTypeUtils::size_of(tensor->dtype()))
                      << "\n";
        }

        const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        const double fp32_mib =
            static_cast<double>(total_params * sizeof(float)) /
            (1024.0 * 1024.0);

        std::cout << "\n========== BF16 LOAD CHECK ==========\n";
        std::cout << "Parameters:       " << total_params << "\n";
        std::cout << "BF16 tensors:     " << bf16_tensors << "\n";
        std::cout << "FP32 tensors:     " << fp32_tensors << "\n";
        std::cout << "Parameter bytes:  " << total_bytes << "\n";
        std::cout << "Parameter MiB:    " << mib << "\n";
        std::cout << "FP32 baseline:    " << fp32_mib << " MiB\n";

        if (bf16_tensors == 0) {
            std::cerr << "[FAIL] No BF16 tensors were produced.\n";
            return 2;
        }

        if (fp32_tensors != 0) {
            std::cerr << "[FAIL] Some parameters are still FP32.\n";
            return 3;
        }

        if (total_bytes >= total_params * sizeof(float)) {
            std::cerr << "[FAIL] Parameter storage was not reduced.\n";
            return 4;
        }

        std::cout << "[PASS] Parameters loaded as BF16.\n";
        std::cout << "[PASS] No FP32 parameter tensors remain in this load result.\n";
        std::cout << "[PASS] Test exits before forward/backward/optimizer.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
