#include "wikitext2_dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ops;

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

    WikiText2Dataset ds(cfg, [](const std::string&) { return std::vector<int32_t>{}; });
    ds.load(Split::Train);
    Batch batch = ds.next_batch(2, false);

    const int32_t* input_ids = batch.input_ids->data<int32_t>();
    const int32_t* labels = batch.labels->data<int32_t>();
    const float* attention_mask = batch.attention_mask->data<float>();

    const int32_t expected_input_ids[10] = {
        10, 11, 12, 99, 99,
        20, 21, 22, 23, 24,
    };
    const int32_t expected_labels[10] = {
        -100, -100, 12, -100, -100,
        -100, 21, -100, 23, 24,
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

    fs::remove(jsonl_path);
    fs::remove(tmp_dir);

    std::cout << "JSONL mask alignment test passed\n";
    return 0;
}
