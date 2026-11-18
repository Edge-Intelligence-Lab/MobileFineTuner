#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "mmlu_dataset.h"
#include "wikitext2_dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ops;

void test_mmlu_answer_alignment(const std::filesystem::path &root) {
    namespace fs = std::filesystem;
    fs::create_directories(root / "dev");
    {
        std::ofstream f(root / "vocab.json");
        f << R"({"<|endoftext|>":0,"A":1,"B":2,"C":3,"D":4,"a":5,"b":6,"c":7,"d":8,"e":9,"Ġ":10})";
    }
    {
        std::ofstream f(root / "merges.txt");
        f << "#version: 0.2\n";
    }
    {
        std::ofstream f(root / "dev/tiny_dev.csv");
        f << "a,b,c,d,e,A\n";
    }
    auto tokenizer_cfg = QwenTokenizerConfig::from_pretrained(root.string());
    tokenizer_cfg.eos_token_id = tokenizer_cfg.pad_token_id = tokenizer_cfg.unk_token_id = 0;
    QwenBPETokenizer tokenizer(tokenizer_cfg);
    tokenizer.load();
    MMLUSample sample;
    sample.question = "a";
    sample.choices[0] = "b";
    sample.choices[1] = "c";
    sample.choices[2] = "d";
    sample.choices[3] = "e";
    sample.answer = 'A';
    const auto prompt = tokenizer.encode(sample.to_prompt());
    const auto full = tokenizer.encode(sample.to_full_text());
    if (full.size() <= prompt.size())
        throw std::runtime_error("MMLU fixture has no answer tokens");
    MMLUConfig cfg;
    cfg.data_dir = root.string();
    cfg.seq_len = 256;
    cfg.batch_size = 1;
    MMLUDataset dataset(cfg, &tokenizer);
    dataset.load_train();
    auto batch = dataset.next_batch(false);
    auto logits = zeros({1, cfg.seq_len, 32});
    logits->set_requires_grad(true);
    for (int i = 0; i < cfg.seq_len; ++i) {
        const int32_t expected =
            size_t(i) >= prompt.size() && size_t(i) < full.size() ? full[i] : -100;
        if (batch.labels->data<int32_t>()[i] != expected)
            throw std::runtime_error("MMLU answers must be unshifted; the loss shifts once");
    }
    auto loss = lm_cross_entropy(logits, batch.labels);
    loss->backward();
    for (int i = 0; i < cfg.seq_len; ++i) {
        bool any_gradient = false;
        for (int v = 0; v < 32; ++v)
            any_gradient |= logits->grad()->data<float>()[i * 32 + v] != 0;
        const bool predicts_answer = size_t(i + 1) >= prompt.size() && size_t(i + 1) < full.size();
        if (any_gradient != predicts_answer)
            throw std::runtime_error("MMLU loss gradient supervises the wrong token");
    }
}

int main() {
    namespace fs = std::filesystem;

    const fs::path tmp_dir = fs::temp_directory_path() / "mf_jsonl_mask_alignment";
    fs::create_directories(tmp_dir);
    const fs::path jsonl_path = tmp_dir / "train.jsonl";

    {
        std::ofstream out(jsonl_path);
        out << R"({"ids":[10,11,12],"mask":[0,0,1]})" << "\n";
        out << R"({"ids":[20,21,22,23,24],"mask":[0,1,0,1,1]})" << "\n";
    }

    WT2Config cfg;
    cfg.jsonl_train = jsonl_path.string();
    cfg.seq_len = 5;
    cfg.pad_id = 99;
    cfg.shuffle_train = false;

    WikiText2Dataset ds(cfg, [](const std::string &) { return std::vector<int32_t>{}; });
    ds.load(Split::Train);
    Batch batch = ds.next_batch(2, false);

    const int32_t *input_ids = batch.input_ids->data<int32_t>();
    const int32_t *labels = batch.labels->data<int32_t>();
    const float *attention_mask = batch.attention_mask->data<float>();

    const int32_t expected_input_ids[10] = {
        10, 11, 12, 99, 99, 20, 21, 22, 23, 24,
    };
    const int32_t expected_labels[10] = {
        -100, -100, 12, -100, -100, -100, 21, -100, 23, 24,
    };

    for (int i = 0; i < 10; ++i) {
        if (input_ids[i] != expected_input_ids[i]) {
            throw std::runtime_error("input_ids mismatch at index " + std::to_string(i));
        }
        if (labels[i] != expected_labels[i]) {
            throw std::runtime_error("labels mismatch at index " + std::to_string(i));
        }
        if (attention_mask[i] != 1.0f) {
            throw std::runtime_error("attention_mask mismatch at index " + std::to_string(i));
        }
    }

    test_mmlu_answer_alignment(tmp_dir / "mmlu");
    fs::remove_all(tmp_dir);

    std::cout << "JSONL and MMLU causal-label alignment tests passed\n";
    return 0;
}
