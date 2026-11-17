#include "../../third_party/nlohmann/json.hpp"
#include "../core/autograd_engine.h"
#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../optim/adam.h"
#include "auto_model.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ops;
namespace fs = std::filesystem;
namespace {
void check(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}
double compare(const TensorPtr& got, const TensorPtr& expected, const std::string& name,
               float atol = 3e-5f, float rtol = 3e-4f) {
    check(got && expected && got->shape() == expected->shape(), name + " shape mismatch");
    double maximum = 0;
    for (int64_t i = 0; i < got->numel(); ++i) {
        const double a = got->data<float>()[i], b = expected->data<float>()[i];
        const double error = std::abs(a - b);
        maximum = std::max(maximum, error);
        if (!std::isfinite(a) || !std::isfinite(b) || error > atol + rtol * std::abs(b))
            throw std::runtime_error(name + " mismatch at " + std::to_string(i) + ": got=" +
                                     std::to_string(a) + " expected=" + std::to_string(b));
    }
    return maximum;
}

void run(const fs::path& path, bool streaming) {
    auto& engine = autograd::Engine::instance();
    engine.clear_graph();
    engine.set_enabled(true);
    std::ifstream protocol_file(path / "protocol.json");
    const auto protocol = nlohmann::json::parse(protocol_file);
    const bool inference_only = protocol.at("inference_only").get<bool>();
    const auto optimizer_kind = protocol.value("optimizer", std::string("adam"));
    check(optimizer_kind == "adam" || optimizer_kind == "sgd", "Unsupported reference optimizer");
    const bool full_size = protocol.value("full_size", inference_only);
    SafeTensorsReader input((path / "inputs.safetensors").string());
    input.parse_header();
    auto ids = input.load_tensor("ids"), mask = input.load_tensor("mask"),
         labels = input.load_tensor("labels");
    SafeTensorsReader ref((path / "reference.safetensors").string());
    ref.parse_header();
    AutoModelLoadOptions load_options;
    const bool native_storage = protocol.value("native_storage", false);
    load_options.safetensors_options.auto_promote_fp16 = !native_storage;
    auto model = AutoModelForCausalLM::from_pretrained((path / "base").string(), load_options);
    auto& qwen = *model->qwen();
    if (native_storage && !full_size)
        check(qwen.embedding_weight()->dtype() == kBFloat16,
              "Native BF16 storage was promoted unexpectedly");
    qwen.load_lora_adapter((path / "initial").string());
    qwen.freeze_base();
    qwen.set_training(false);
    qwen.set_memory_efficient_attention(streaming);
    auto logits = model->forward(ids, mask);
    // Full-size FP32 GEMM reductions accumulate more error than tiny cases.
    double logit_error = compare(logits, ref.load_tensor("logits"), "logits",
                                 full_size ? 1e-3f : 3e-5f, full_size ? 1e-4f : 3e-4f);
    auto loss = lm_cross_entropy(logits, labels, -100, "mean");
    compare(loss->reshape({1}), ref.load_tensor("loss"), "loss", full_size ? 2e-4f : 3e-6f);
    double grad_error = 0, update_error = 0;
    if (!inference_only) {
        loss->backward();
        auto named = qwen.named_lora_parameters();
        for (const auto& item : named) {
            check(bool(item.second->grad()), "missing gradient: " + item.first);
            auto actual = transpose(item.second->grad(), 0, 1);
            auto reference = ref.load_tensor("grad." + item.first);
            if (full_size) {
                // Deep full-size FP32 graphs use different BLAS reduction orders.
                // Check each tensor's relative L2 and worst error against its scale,
                // so cancellation around zero does not hide a globally wrong gradient.
                double squared_error = 0, squared_ref = 0, maximum = 0, reference_max = 0;
                check(actual->shape() == reference->shape(), "Gradient shape mismatch");
                for (int64_t i = 0; i < actual->numel(); ++i) {
                    const double a = actual->data<float>()[i], b = reference->data<float>()[i];
                    check(std::isfinite(a) && std::isfinite(b), "Non-finite full-model gradient");
                    squared_error += (a - b) * (a - b);
                    squared_ref += b * b;
                    maximum = std::max(maximum, std::abs(a - b));
                    reference_max = std::max(reference_max, std::abs(b));
                }
                check(std::sqrt(squared_error) <= 5e-4 * std::sqrt(squared_ref) + 1e-7,
                      "Full-model gradient relative L2: " + item.first);
                check(maximum <= 1e-3 * reference_max + 1e-6,
                      "Full-model gradient maximum error: " + item.first);
                grad_error = std::max(grad_error, maximum);
            } else
                grad_error = std::max(
                    grad_error, compare(actual, reference, "gradient " + item.first, 2e-6f, 3e-3f));
        }

        Adam optimizer(AdamConfig(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f));
        auto params = qwen.get_lora_parameters();
        std::vector<TensorPtr> grads;
        for (const auto& p : params)
            grads.push_back(p->grad());
        if (optimizer_kind == "sgd") {
            for (size_t p = 0; p < params.size(); ++p)
                for (int64_t i = 0; i < params[p]->numel(); ++i)
                    params[p]->data<float>()[i] -= 1e-3f * grads[p]->data<float>()[i];
        } else
            optimizer.step(params, grads);
        for (const auto& item : named)
            update_error =
                std::max(update_error, compare(transpose(item.second, 0, 1),
                                               ref.load_tensor("after." + item.first),
                                               "Optimizer update " + item.first, 2e-6f, 3e-4f));
        engine.clear_graph();
        engine.set_enabled(false);
        auto expected = model->forward(ids, mask);
        compare(expected, ref.load_tensor("after_logits"), "post-update logits",
                full_size ? 1e-3f : 3e-5f, full_size ? 1e-4f : 3e-4f);
        const auto saved = path / (streaming ? "native_streamed" : "native_standard");
        qwen.save_lora_adapter(saved.string());
        qwen.load_lora_adapter(saved.string());
        qwen.freeze_base();
        qwen.set_training(false);
        compare(model->forward(ids, mask), expected, "native adapter reload", 1e-7f, 1e-6f);
        // Load a post-update PEFT checkpoint into the same native base.
        qwen.load_lora_adapter((path / "after").string());
        qwen.freeze_base();
        qwen.set_training(false);
        compare(model->forward(ids, mask), expected, "PEFT post-update adapter",
                full_size ? 1e-3f : 3e-5f, full_size ? 1e-4f : 3e-4f);
        const auto before_bad_load = model->forward(ids, mask);
        const auto bad = path / "invalid_adapter";
        fs::create_directories(bad);
        std::ifstream config_file(path / "initial/adapter_config.json");
        auto config = nlohmann::json::parse(config_file);
        config["r"] = config["r"].get<int>() + 1;
        {
            std::ofstream file(bad / "adapter_config.json");
            file << config.dump();
        }
        fs::copy_file(path / "initial/adapter_model.safetensors", bad / "adapter_model.safetensors",
                      fs::copy_options::overwrite_existing);
        bool rejected = false;
        try {
            qwen.load_lora_adapter(bad.string());
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected, "Incompatible adapter rank was accepted");
        compare(model->forward(ids, mask), before_bad_load,
                "failed adapter load changed active model", 1e-7f, 1e-6f);
    }
    std::cout << path.filename().string() << " streaming=" << streaming
              << " logits_max_abs=" << logit_error << " grad_max_abs=" << grad_error
              << " optimizer=" << optimizer_kind << " update_max_abs=" << update_error << " PASS\n";
    engine.clear_graph();
    engine.set_enabled(true);
}
} // namespace
int main(int argc, char** argv) try {
    const char* root = argc > 1 ? argv[1] : std::getenv("MFT_QWEN_HF_FIXTURES");
    if (!root) {
        std::cout << "Set MFT_QWEN_HF_FIXTURES after generating references with "
                     "scripts/generate_qwen_hf_fixtures.py\n";
        return 77;
    }
    check(fs::is_directory(root), "HF fixture directory does not exist");
    int cases = 0;
    for (const auto& entry : fs::directory_iterator(root))
        if (fs::exists(entry.path() / "protocol.json")) {
            run(entry.path(), false);
            run(entry.path(), true);
            ++cases;
        }
    check(cases > 0, "No Qwen HF fixtures found");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
