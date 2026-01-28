#include "gemma_model.h"
#include "safetensors_loader.h"
#include "../core/tokenizer_gemma.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

using namespace ops;

struct GemmaGoldenSample {
    std::string text;
    std::vector<int32_t> input_ids;
    std::vector<float> logits;
};

bool load_golden(const std::string& path, std::vector<GemmaGoldenSample>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open golden file: " << path << std::endl;
        return false;
    }

    uint32_t num_samples = 0;
    f.read(reinterpret_cast<char*>(&num_samples), sizeof(uint32_t));
    if (!f) return false;

    out.clear();
    out.reserve(num_samples);
    for (uint32_t i = 0; i < num_samples; ++i) {
        GemmaGoldenSample sample;
        uint32_t text_len = 0;
        f.read(reinterpret_cast<char*>(&text_len), sizeof(uint32_t));
        if (!f) return false;
        sample.text.resize(text_len);
        f.read(sample.text.data(), text_len);

        uint32_t seq_len = 0;
        f.read(reinterpret_cast<char*>(&seq_len), sizeof(uint32_t));
        sample.input_ids.resize(seq_len);
        f.read(reinterpret_cast<char*>(sample.input_ids.data()), seq_len * sizeof(int32_t));

        uint32_t vocab = 0;
        f.read(reinterpret_cast<char*>(&vocab), sizeof(uint32_t));
        sample.logits.resize(vocab);
        f.read(reinterpret_cast<char*>(sample.logits.data()), vocab * sizeof(float));

        if (!f) return false;
        out.push_back(std::move(sample));
    }
    return true;
}

TensorPtr tensor_from_ids(const std::vector<int32_t>& ids) {
    auto tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(ids.size())},
        DType::kInt32,
        kCPU);
    std::memcpy(tensor->data<int32_t>(), ids.data(), ids.size() * sizeof(int32_t));
    return tensor;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <golden_bin>" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string golden_path = argv[2];

    std::vector<GemmaGoldenSample> samples;
    if (!load_golden(golden_path, samples)) {
        return 1;
    }

    auto cfg = GemmaTextConfig::from_pretrained(model_dir);
    GemmaModel model(cfg);

    SafeTensorsReader reader(model_dir + "/model.safetensors");
    reader.parse_header();
    auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);
    SafeTensorsLoadOptions opts;
    opts.verbose = false;
    auto tensors = reader.load_tensors_mapped(mapping, opts);
    for (auto& kv : tensors) {
        model.assign_weight(kv.first, kv.second);
    }

    auto tok_cfg = GemmaTokenizerConfig::from_pretrained(model_dir);
    GemmaTokenizer tokenizer(tok_cfg);
    tokenizer.load();

    const double tolerance = 5e-4;
    double global_max_err = 0.0;
    double global_mean_err = 0.0;
    int count = 0;

    for (const auto& sample : samples) {
        auto encoded = tokenizer.encode(sample.text, false, 0, false);
        if (encoded != std::vector<int>(sample.input_ids.begin(), sample.input_ids.end())) {
            std::cerr << "Tokenizer mismatch for text: " << sample.text << std::endl;
            return 1;
        }

        auto input = tensor_from_ids(sample.input_ids);
        auto logits = model.forward(input);
        const auto& shape = logits->shape();
        int64_t seq_len = shape[1];
        int64_t vocab = shape[2];
        const float* data = logits->data<float>();
        const float* last = data + (seq_len - 1) * vocab;

        double max_err = 0.0;
        double sum_err = 0.0;
        for (int64_t i = 0; i < vocab; ++i) {
            double diff = std::fabs(last[i] - sample.logits[i]);
            if (diff > max_err) max_err = diff;
            sum_err += diff;
        }
        double mean_err = sum_err / static_cast<double>(vocab);

        std::cout << "Sample \"" << sample.text << "\" max_err=" << max_err
                  << " mean_err=" << mean_err << std::endl;

        global_max_err = std::max(global_max_err, max_err);
        global_mean_err += mean_err;
        count++;

        if (max_err > tolerance) {
            std::cerr << "Mismatch exceeds tolerance." << std::endl;
            return 1;
        }
    }

    global_mean_err /= std::max(1, count);
    std::cout << "All samples matched. Global max_err=" << global_max_err
              << " avg_mean_err=" << global_mean_err << std::endl;
    return 0;
}
