// Standalone regression: link this file against operators after core integration.
// No model, tokenizer download, network, or external dataset is required.
#include "adam.h"
#include "../data/wikitext2_dataset.h"
#include "../data/mmlu_dataset.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
using namespace ops;
using Json = nlohmann::json;

void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

template <class F> void rejects(F&& f, const std::string& message) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected rejection: " + message);
}

TensorPtr values(float start = 1.0f) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{3}, kFloat32, kCPU);
    for (int i = 0; i < 3; ++i) tensor->data<float>()[i] = start + i * 0.125f;
    return tensor;
}

void equal_tensor(const TensorPtr& a, const TensorPtr& b) {
    check(static_cast<bool>(a) == static_cast<bool>(b), "tensor presence differs");
    if (!a) return;
    check(a->shape() == b->shape() && a->dtype() == b->dtype(), "tensor metadata differs");
    check(std::memcmp(a->data_ptr(), b->data_ptr(), static_cast<size_t>(a->numel()) * 4) == 0,
          "tensor values are not bitwise equal");
}

AdamSnapshot owned_snapshot(const AdamSnapshot& source) {
    AdamSnapshot result{source.metadata, {}};
    for (const auto& [name, tensor] : source.tensors) result.tensors[name] = tensor->clone();
    return result;
}

Json normalized_metadata(Json j) {
    auto& params = j["parameters"];
    std::sort(params.begin(), params.end(), [](const Json& a, const Json& b) {
        return a.at("name").get<std::string>() < b.at("name").get<std::string>();
    });
    return j;
}

void equal_snapshot(const AdamSnapshot& a, const AdamSnapshot& b) {
    check(normalized_metadata(a.metadata) == normalized_metadata(b.metadata), "optimizer metadata differs");
    check(a.tensors.size() == b.tensors.size(), "optimizer tensor count differs");
    for (const auto& [name, tensor] : a.tensors) equal_tensor(tensor, b.tensors.at(name));
}

std::vector<TensorPtr> pointers(const NamedParameters& parameters) {
    std::vector<TensorPtr> result;
    for (const auto& p : parameters) result.push_back(p.second);
    return result;
}

void test_adam(bool adamw, bool amsgrad) {
    AdamWConfig config(0.017f, 0.72f, 0.91f, 1e-6f, 0.03f, 0.37f, amsgrad);
    config.decoupled_weight_decay = adamw;
    std::unique_ptr<Adam> live = adamw ? std::make_unique<AdamW>(config) : std::make_unique<Adam>(config);
    NamedParameters parameters{{"layer/a", values()}, {"layer/b", values(-0.5f)}, {"unused", values(0.3f)}};
    auto ptrs = pointers(parameters);
    // Distinct per-parameter steps and one never-initialized/no-gradient tensor.
    live->step(ptrs, {values(0.1f), nullptr, nullptr});
    live->step(ptrs, {values(-0.3f), values(0.4f), nullptr});
    live->set_learning_rate(0.009f);
    auto borrowed = live->export_named_state(parameters);
    auto borrowed_again = live->export_named_state(parameters);
    for (const auto& [name, tensor] : borrowed.tensors)
        check(tensor == borrowed_again.tensors.at(name), "export unexpectedly cloned storage");
    check(borrowed.metadata["parameters"][2]["initialized"] == false, "uninitialized state lost");
    check(borrowed.metadata["config"]["clip_grad_norm"] == config.clip_grad_norm, "clip config lost");
    check(borrowed.metadata["parameters"][0]["step"] == 2 &&
          borrowed.metadata["parameters"][1]["step"] == 1, "per-parameter steps lost");
    auto disk = owned_snapshot(borrowed);
    disk.metadata = Json::parse(disk.metadata.dump());
    NamedParameters fresh;
    for (const auto& [name, tensor] : parameters) {
        fresh.emplace_back(name, tensor->clone());
        check(tensor.get() != fresh.back().second.get() && tensor->data_ptr() != fresh.back().second->data_ptr(),
              "test must use fresh parameter addresses");
    }
    Adam resumed{AdamConfig()};
    auto reversed = fresh;
    std::reverse(reversed.begin(), reversed.end());
    resumed.import_named_state(reversed, disk);
    equal_snapshot(live->export_named_state(parameters), resumed.export_named_state(reversed));
    for (const auto& [name, tensor] : disk.tensors)
        check(tensor == resumed.export_named_state(fresh).tensors.at(name), "import did not adopt storage");
    check(resumed.get_config().clip_grad_norm == config.clip_grad_norm, "base config was not restored");
    auto fresh_ptrs = pointers(fresh);
    for (int i = 0; i < 4; ++i) {
        std::vector<TensorPtr> gradients{values(0.02f * i), i == 1 ? nullptr : values(-0.1f),
                                         i == 0 ? nullptr : values(0.2f)};
        live->step(ptrs, gradients);
        resumed.step(fresh_ptrs, gradients);
        for (size_t p = 0; p < ptrs.size(); ++p) equal_tensor(ptrs[p], fresh_ptrs[p]);
        equal_snapshot(live->export_named_state(parameters), resumed.export_named_state(fresh));
    }
    Adam empty{config};
    auto empty_snapshot = empty.export_named_state(fresh);
    resumed.import_named_state(fresh, empty_snapshot);
    check(resumed.export_named_state(fresh).tensors.empty(), "uninitialized import failed to clear old state");
    empty.step(fresh_ptrs, {nullptr, nullptr, nullptr});
    equal_snapshot(empty_snapshot, empty.export_named_state(fresh));
}

void test_adam_corruption() {
    Adam source{AdamConfig(0.01f, 0.9f, 0.99f, 1e-7f, 0.02f, 0.5f, true)};
    NamedParameters named{{"a", values()}, {"b", values(2)}, {"idle", values(3)}};
    source.step(pointers(named), {values(0.1f), values(0.2f), nullptr});
    const auto good = owned_snapshot(source.export_named_state(named));
    const auto baseline = owned_snapshot(good);
    auto unchanged = [&]() { equal_snapshot(baseline, source.export_named_state(named)); };
    std::vector<std::function<void(AdamSnapshot&)>> mutations{
        [](auto& s) { s.metadata["version"] = 2; },
        [](auto& s) { s.metadata["config"]["learning_rate"] = -1; },
        [](auto& s) { s.metadata["config"]["beta1"] = 1; },
        [](auto& s) { s.metadata["config"]["epsilon"] = 0; },
        [](auto& s) { s.metadata["config"]["clip_grad_norm"] = std::numeric_limits<double>::infinity(); },
        [](auto& s) { s.metadata["config"]["amsgrad"] = 1; },
        [](auto& s) { s.metadata["config"].erase("decoupled_weight_decay"); },
        [](auto& s) { s.metadata["extra"] = true; },
        [](auto& s) { s.metadata["parameters"][1]["name"] = "a"; },
        [](auto& s) { s.metadata["parameters"][1]["name"] = "unknown"; },
        [](auto& s) { s.metadata["parameters"].erase(1); },
        [](auto& s) { s.metadata["parameters"][1]["shape"] = Json::array({4}); },
        [](auto& s) { s.metadata["parameters"][1]["shape"] = Json::array({3.0}); },
        [](auto& s) { s.metadata["parameters"][1]["dtype"] = "float16"; },
        [](auto& s) { s.metadata["parameters"][1]["step"] = -1; },
        [](auto& s) { s.metadata["parameters"][1]["step"] = 1.5; },
        [](auto& s) { s.metadata["parameters"][1]["step"] = std::numeric_limits<uint64_t>::max(); },
        [](auto& s) { s.metadata["parameters"][2]["step"] = 1; },
        [](auto& s) { s.tensors.erase("vhat/b"); },
        [](auto& s) { s.tensors["surprise"] = values(); },
        [](auto& s) { s.tensors["m/idle"] = values(); },
        [](auto& s) { s.tensors["m/b"] = nullptr; },
        [](auto& s) { s.tensors["m/b"] = std::make_shared<Tensor>(std::vector<int64_t>{1, 3}); },
        [](auto& s) { s.tensors["m/b"] = std::make_shared<Tensor>(std::vector<int64_t>{3}, kInt32); },
        [](auto& s) { s.tensors["v/b"]->template data<float>()[2] = -1; },
        [](auto& s) { s.tensors["m/b"]->template data<float>()[2] = std::numeric_limits<float>::quiet_NaN(); },
        [](auto& s) { s.tensors["vhat/b"]->template data<float>()[2] = std::numeric_limits<float>::infinity(); },
        [](auto& s) { s.tensors["v/b"] = s.tensors.at("m/a"); },
        [&](auto& s) { s.tensors["m/b"] = named[0].second; },
        [&](auto& s) { s.tensors["m/b"] = from_blob(named[0].second->data<float>(), {3}, kFloat32, kCPU); }
    };
    for (const auto& mutate : mutations) {
        auto bad = owned_snapshot(good);
        mutate(bad);
        // A valid changed config must not be committed before a later error.
        if (bad.metadata["config"]["learning_rate"] == good.metadata["config"]["learning_rate"])
            bad.metadata["config"]["learning_rate"] = 0.123f;
        rejects([&] { source.import_named_state(named, bad); }, "corrupt optimizer");
        unchanged();
    }
    for (int variant = 0; variant < 4; ++variant) {
        auto bad_names = named;
        if (variant == 0) bad_names[1].first = "a";
        if (variant == 1) bad_names[1].first.clear();
        if (variant == 2) bad_names[1].second = bad_names[0].second;
        if (variant == 3) bad_names.pop_back();
        rejects([&] { source.import_named_state(bad_names, good); }, "bad caller mapping");
        unchanged();
    }
    rejects([&] { source.export_named_state({named[0], named[2]}); }, "omitted initialized parameter");
    auto overlap = std::make_shared<Tensor>(std::vector<int64_t>{6});
    NamedParameters aliased{{"x", Tensor::make_view(overlap, {3}, {1}, 0)},
                            {"y", Tensor::make_view(overlap, {3}, {1}, 1)}};
    rejects([&] { source.import_named_state(aliased, good); }, "overlapping parameter views");
    unchanged();
}

struct Fixtures {
    std::filesystem::path root;
    Fixtures() {
        std::random_device random;
        bool created = false;
        for (int attempt = 0; attempt < 32 && !created; ++attempt) {
            root = std::filesystem::temp_directory_path() /
                   ("mft-resume-" + std::to_string(random()) + "-" + std::to_string(random()));
            created = std::filesystem::create_directory(root);
        }
        check(created, "cannot create temporary fixtures");
        std::ofstream text(root / "tokens.txt");
        for (int i = 1; i <= 53; ++i) text << i << (i % 7 ? ' ' : '\n');
        text << '\n';
        std::ofstream bin(root / "tokens.bin", std::ios::binary);
        for (int32_t i = 1; i <= 53; ++i) bin.write(reinterpret_cast<const char*>(&i), sizeof(i));
        std::ofstream meta(root / "meta.json");
        meta << R"({"total_tokens":53,"eos_token_id":99,"pad_token_id":0,"train":{"offset":0,"length":53},"valid":{"offset":0,"length":53},"test":{"offset":0,"length":53}})";
        std::ofstream jsonl(root / "samples.jsonl");
        for (int i = 1; i <= 7; ++i) {
            Json ids = Json::array({i, i + 10, i + 20});
            Json mask = Json::array({0, 1, 1});
            if (i % 2 == 0) { ids.push_back(i + 30); mask.push_back(1); }
            jsonl << Json{{"ids", ids}, {"mask", mask}}.dump() << '\n';
        }
        for (const char* split : {"auxiliary_train", "val", "test"}) {
            std::filesystem::create_directories(root / split);
            std::ofstream csv(root / split / (std::string("tiny_") + split + ".csv"));
            for (int i = 0; i < 7; ++i)
                csv << "question" << i << ",one,two,three,four," << static_cast<char>('A' + i % 4) << '\n';
        }
        check(text.good() && bin.good() && meta.good() && jsonl.good(), "fixture write failed");
    }
    ~Fixtures() { std::error_code error; std::filesystem::remove_all(root, error); }
};

std::vector<int32_t> numeric_encode(const std::string& text) {
    std::istringstream in(text);
    std::vector<int32_t> ids;
    int32_t value;
    while (in >> value) ids.push_back(value);
    return ids;
}

std::vector<int32_t> byte_encode(const std::string& text) {
    return std::vector<int32_t>(text.begin(), text.end());
}

template <class B> void equal_batch(const B& a, const B& b) {
    equal_tensor(a.input_ids, b.input_ids);
    equal_tensor(a.attention_mask, b.attention_mask);
    equal_tensor(a.labels, b.labels);
}

template <class Dataset> void dataset_corruption(Dataset& dataset) {
    const auto before = dataset.export_state();
    const auto original = Json::parse(before);
    std::vector<std::function<void(Json&)>> mutations{
        [](auto& j) { j["schema"] = "wrong"; },
        [](auto& j) { j["version"] = 2; },
        [](auto& j) { j["version"] = 1.0; },
        [](auto& j) { j["dataset"] = "wrong"; },
        [](auto& j) { j["split"] = "wrong"; },
        [](auto& j) { j["mode"] = "wrong"; },
        [](auto& j) { j["cursor_unit"] = "wrong"; },
        [](auto& j) { j["cursor"] = -1; },
        [](auto& j) { j["cursor"] = 0.5; },
        [](auto& j) { j["cursor"] = std::numeric_limits<uint64_t>::max(); },
        [](auto& j) { j["order"][1] = j["order"][0]; },
        [](auto& j) { j["order"][0] = j["order"].size(); },
        [](auto& j) { j["order"][0] = -1; },
        [](auto& j) { j["order"][0] = 0.0; },
        [](auto& j) { j["order"].erase(0); },
        [](auto& j) { j["config"]["seq_len"] = 999; },
        [](auto& j) { j["config"]["seed"] = nullptr; },
        [](auto& j) { j["config"]["extra"] = true; },
        [](auto& j) { j["rng"] = "1 2"; },
        [](auto& j) { std::string zeros = "0"; for (size_t i = 1; i < std::mt19937_64::state_size; ++i) zeros += " 0"; j["rng"] = zeros; },
        [](auto& j) { j["rng"] = j["rng"].template get<std::string>() + " garbage"; },
        [](auto& j) { j["rng_engine"] = "wrong"; },
        [](auto& j) { j.erase("rng"); },
        [](auto& j) { j["extra"] = true; }
    };
    if (original.contains("starts")) {
        mutations.push_back([](auto& j) { j["starts"].push_back(999); });
        mutations.push_back([](auto& j) { j["sequence_count"] = 0; });
        mutations.push_back([](auto& j) { j["token_count"] = 0; });
    } else {
        mutations.push_back([](auto& j) { j["sample_count"] = 0; });
        mutations.push_back([](auto& j) { j["batch_count"] = 0; });
        mutations.push_back([](auto& j) { j["sample_index_fnv1a64"] = "0"; });
    }
    for (const auto& mutate : mutations) {
        auto bad = original;
        mutate(bad);
        rejects([&] { dataset.import_state(bad.dump()); }, "corrupt dataset state");
        check(before == dataset.export_state(), "dataset mutated after rejected import");
    }
    for (const auto& bad : {before.substr(0, before.size() / 2),
                            std::string("{\"cursor\":0,") + before.substr(1), before + " trailing"}) {
        rejects([&] { dataset.import_state(bad); }, "malformed/duplicate JSON");
        check(before == dataset.export_state(), "malformed JSON mutated dataset");
    }
}

void test_wikitext(const Fixtures& files) {
    for (int mode = 0; mode < 4; ++mode) for (bool shuffle : {false, true})
    for (bool drop : {false, true}) for (size_t batch_size : {1u, 3u, 100u}) {
        WT2Config config;
        config.seq_len = 4;
        config.stride = 3;
        config.eos_id = 99;
        config.insert_eos_between_lines = false;
        config.drop_last = drop;
        config.shuffle_train = shuffle;
        config.streaming_mode = mode == 1;
        config.max_cache_tokens = 9;
        config.train_path = config.valid_path = config.test_path = (files.root / "tokens.txt").string();
        if (mode == 2) config.pretokenized_path = (files.root / "tokens.bin").string();
        if (mode == 3) config.jsonl_train = config.jsonl_valid = config.jsonl_test = (files.root / "samples.jsonl").string();
        WikiText2Dataset live(config, numeric_encode), fresh(config, numeric_encode);
        rejects([&] { live.export_state(); }, "unloaded WikiText");
        live.load(Split::Train);
        fresh.load(Split::Train);
        live.next_batch(batch_size, false);
        auto state = live.export_state();
        fresh.shuffle();  // Restore must replace RNG and order, not just cursor.
        fresh.import_state(state);
        check(state == fresh.export_state(), "WikiText roundtrip failed");
        auto held = fresh.get_batch(0, 1);
        auto held_ids = held.input_ids->clone();
        fresh.import_state(state);
        equal_tensor(held.input_ids, held_ids);
        for (int i = 0; i < 30; ++i) {
            bool loop = i >= 15;
            equal_batch(live.next_batch(batch_size, loop), fresh.next_batch(batch_size, loop));
            check(live.export_state() == fresh.export_state(), "WikiText traversal/RNG drift");
            if (i == 10 || i == 20) fresh.import_state(live.export_state());
        }
        // Compare the next epoch shuffle as well as the current permutation.
        live.shuffle(); fresh.shuffle();
        check(live.export_state() == fresh.export_state(), "WikiText next shuffle differs");
        dataset_corruption(fresh);
        // End cursor is legal; no-loop must not consume or reshuffle at end.
        live.reset_cursor();
        while (live.next_batch(1, false).input_ids) {}
        fresh.import_state(live.export_state());
        check(!fresh.next_batch(1, false).input_ids, "WikiText end cursor not restored");
        equal_batch(live.next_batch(1, true), fresh.next_batch(1, true));
        fresh.load(Split::Valid);
        const auto valid_state = fresh.export_state();
        rejects([&] { fresh.import_state(state); }, "WikiText split mismatch");
        check(valid_state == fresh.export_state(), "split mismatch mutated WikiText");
        fresh.load(Split::Test);
        auto test_state = fresh.export_state();
        fresh.next_batch(1, false);
        fresh.import_state(test_state);
        check(test_state == fresh.export_state(), "WikiText test split roundtrip failed");
    }
}

void test_mmlu(const Fixtures& files) {
    for (int batch_size : {1, 3, 20}) for (bool shuffle : {false, true}) {
        MMLUConfig config;
        config.data_dir = files.root.string();
        config.batch_size = batch_size;
        config.seq_len = 128;
        MMLUDataset live(config, byte_encode), fresh(config, byte_encode);
        rejects([&] { fresh.export_state(); }, "unloaded MMLU");
        live.load_train(); fresh.load_train();
        if (shuffle) live.shuffle();
        live.next_batch(false);
        auto state = live.export_state();
        check(Json::parse(state)["cursor"] == 1, "MMLU cursor must count batches");
        fresh.shuffle();
        fresh.import_state(state);
        check(state == fresh.export_state(), "MMLU roundtrip failed");
        for (int i = 0; i < 20; ++i) {
            auto a = live.next_batch(i >= 10), b = fresh.next_batch(i >= 10);
            check(a.num_samples == b.num_samples, "MMLU batch length differs");
            equal_batch(a, b);
            check(live.export_state() == fresh.export_state(), "MMLU traversal drift");
            if (i == 7 || i == 12) fresh.import_state(live.export_state());
        }
        live.shuffle(); fresh.shuffle();
        check(live.export_state() == fresh.export_state(), "MMLU next shuffle differs");
        dataset_corruption(fresh);
        live.reset_cursor();
        while (live.next_batch(false).num_samples) {}
        fresh.import_state(live.export_state());
        check(fresh.next_batch(false).num_samples == 0, "MMLU end cursor not restored");
        equal_batch(live.next_batch(true), fresh.next_batch(true));
        auto different_config = config;
        ++different_config.batch_size;
        MMLUDataset mismatch(different_config, byte_encode);
        mismatch.load_train();
        rejects([&] { mismatch.import_state(state); }, "MMLU batch config mismatch");
        fresh.load_val();
        const auto val_state = fresh.export_state();
        rejects([&] { fresh.import_state(state); }, "MMLU split mismatch");
        check(val_state == fresh.export_state(), "split mismatch mutated MMLU");
        fresh.load_test();
        auto test_state = fresh.export_state();
        fresh.next_batch(false);
        fresh.import_state(test_state);
        check(test_state == fresh.export_state(), "MMLU test split roundtrip failed");
    }
}
}  // namespace

int main() {
    try {
        for (bool adamw : {false, true}) for (bool amsgrad : {false, true}) test_adam(adamw, amsgrad);
        test_adam_corruption();
        Fixtures fixtures;
        test_wikitext(fixtures);
        test_mmlu(fixtures);
        std::cout << "resume state regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resume state regression failed: " << error.what() << '\n';
        return 1;
    }
}
