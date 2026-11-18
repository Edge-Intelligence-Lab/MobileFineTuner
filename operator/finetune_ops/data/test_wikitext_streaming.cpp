#include "wikitext2_dataset.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace ops;
namespace fs = std::filesystem;
namespace {
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
std::vector<int32_t> encode(const std::string &line) {
    std::vector<int32_t> ids;
    // Token counts deliberately differ from file byte offsets.
    for (size_t i = 0; i < line.size(); i += 2) {
        const int first = static_cast<unsigned char>(line[i]) + 1;
        const int second = i + 1 < line.size() ? static_cast<unsigned char>(line[i + 1]) + 1 : 0;
        ids.push_back(first * 257 + second);
    }
    return ids;
}
void run_case(const fs::path &root, bool eos, bool shuffled, float fraction) {
    std::vector<std::string> lines;
    for (int i = 0; i < 1000; ++i)
        lines.push_back(i % 9 == 0 ? "" : "record " + std::to_string(i) + " sample text");
    const auto file = root / "train.raw";
    {
        std::ofstream f(file);
        for (const auto &line : lines)
            f << line << '\n';
    }
    WT2Config cfg;
    cfg.train_path = file.string();
    cfg.valid_path = file.string();
    cfg.seq_len = 8;
    cfg.max_cache_tokens = 33;
    cfg.insert_eos_between_lines = eos;
    cfg.shuffle_train = shuffled;
    cfg.data_fraction = fraction;
    cfg.eos_id = 500;
    cfg.pad_id = 501;
    size_t calls = 0;
    WikiText2Dataset dataset(cfg, [&](const std::string &line) {
        ++calls;
        return encode(line);
    });
    dataset.load(Split::Train);
    std::vector<int32_t> packed;
    for (const auto &line : lines) {
        if (!eos && line.empty())
            continue;
        auto ids = encode(line);
        packed.insert(packed.end(), ids.begin(), ids.end());
        if (eos)
            packed.push_back(cfg.eos_id);
    }
    // Match the existing streaming framing, including its terminal counted slot.
    const size_t counted = packed.size() + 1;
    const size_t limit =
        std::min(counted, std::max(size_t(cfg.seq_len + 1), size_t(double(counted) * fraction)));
    std::vector<size_t> starts;
    for (size_t i = 0; i + cfg.seq_len + 1 <= limit; i += cfg.seq_len)
        starts.push_back(i);
    std::vector<size_t> order(starts.size());
    std::iota(order.begin(), order.end(), 0);
    if (shuffled) {
        std::mt19937_64 rng(cfg.seed);
        std::shuffle(order.begin(), order.end(), rng);
    }
    check(dataset.num_sequences() == starts.size(), "Chunk count changed");
    for (size_t index :
         {size_t(0), starts.size() - 1, starts.size() / 2, size_t(3), starts.size() - 2}) {
        const size_t before = calls;
        auto batch = dataset.get_batch(index, 1);
        check(calls - before <= 8, "A random window re-tokenized the preceding file");
        const size_t start = starts[order[index]];
        const size_t available =
            start < std::min(packed.size(), limit)
                ? std::min(size_t(cfg.seq_len + 1), std::min(packed.size(), limit) - start)
                : 0;
        const size_t length = available >= 2 ? available - 1 : 0;
        for (int i = 0; i < cfg.seq_len; ++i) {
            const bool valid = size_t(i) < length;
            check(batch.input_ids->data<int32_t>()[i] == (valid ? packed[start + i] : cfg.pad_id),
                  "Input IDs changed");
            check(batch.labels->data<int32_t>()[i] == (valid ? packed[start + i] : -100),
                  "Labels changed");
            check(batch.attention_mask->data<float>()[i] == (valid ? 1.0f : 0.0f), "Mask changed");
        }
    }
    // Loading a second split resets and rebuilds the seek index.
    dataset.load(Split::Valid);
    const size_t before = calls;
    auto batch = dataset.get_batch(starts.size() - 1, 1);
    check(calls - before <= 8 && bool(batch.input_ids), "Reloaded split lost indexed lookup");
}
} // namespace
int main() try {
    auto root = fs::temp_directory_path() /
                ("mft-wt2-stream-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        for (bool eos : {false, true})
            for (bool shuffled : {false, true})
                for (float fraction : {0.5f, 1.0f})
                    run_case(root, eos, shuffled, fraction);
    } catch (...) {
        fs::remove_all(root);
        throw;
    }
    fs::remove_all(root);
    std::cout << "PASS: indexed streaming windows preserve IDs/labels/masks/order and avoid prefix "
                 "tokenization\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
