#include "../../opt_ops/sharding/parameter_sharder.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ops;
using namespace ops::sharding;
namespace fs = std::filesystem;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Fn> void throws(Fn fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected exception");
}
void equal(const TensorPtr& a, const TensorPtr& b) {
    check(a && b && a->shape() == b->shape() &&
          std::memcmp(a->data_ptr(), b->data_ptr(), a->numel() * sizeof(float)) == 0,
          "offload values changed");
}
struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("mft-offload-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { fs::create_directory(path); }
    ~TemporaryDirectory() { std::error_code ignored; fs::remove_all(path, ignored); }
};

void roundtrip(const fs::path& root) {
    auto owner = tensor({1.25f, -0.0f, -2.5f, 9.0f});
    auto alias = owner;
    auto second = tensor({2.0f, 3.0f, 4.0f, 5.0f});
    ParameterSharder sharder({root.string(), 16, false});
    sharder.register_parameter("a/b", owner, true, &owner);
    check(owner == alias, "keep_in_memory must not replace source");
    sharder.register_parameter("a_b", second, false, &second);
    check(!second, "initial offload must clear owner");
    check(sharder.stats().writes == 2 && sharder.stats().loads == 0, "registration counters");
    auto b = sharder.require("a_b");
    check(!owner && second == b, "LRU eviction/owner refill failed");
    equal(alias, tensor({1.25f, -0.0f, -2.5f, 9.0f}));
    auto loaded = sharder.require("a/b");
    equal(loaded, alias);
    check(!second && owner == loaded, "owner synchronization failed");
    loaded->data<float>()[2] = 7.5f;
    sharder.mark_dirty("a/b");
    sharder.evict("a/b");
    sharder.evict("a/b"); // Idempotent and not counted twice.
    check(!owner && sharder.stats().resident_bytes == 0, "logical offload failed");
    equal(sharder.require("a/b"), loaded);
    check(sharder.stats().writes == 3 && sharder.stats().loads == 3 &&
          sharder.stats().evictions == 3 && sharder.stats().bytes_read == 48 &&
          sharder.stats().bytes_written == 48, "I/O accounting incorrect");
    sharder.offload_all();
    check(sharder.stats().resident_bytes == 0 && alias->data<float>()[2] == -2.5f,
          "logical stats must not invalidate surviving aliases");
    // Separate instances and names that sanitize identically must never overwrite.
    ParameterSharder other({root.string(), 16, false});
    auto newer = tensor({8.0f, 8.0f, 8.0f, 8.0f});
    other.register_parameter("a/b", newer, false);
    equal(other.require("a/b"), newer);
    equal(sharder.require("a/b"), loaded);
    auto transposed = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, alias->data_ptr())->transpose(0, 1);
    other.register_parameter("transpose", transposed, false);
    equal(other.require("transpose"), transposed);
}

void failures(const fs::path& root) {
    auto owner = tensor({1.0f, 2.0f});
    auto expected = owner;
    ParameterSharder sharder({root.string(), 8, false});
    auto directory = fs::path(sharder.storage_directory());
    fs::remove(directory);
    { std::ofstream blocker(directory); blocker << 'x'; }
    throws([&] { sharder.register_parameter("p", owner, false, &owner); });
    check(owner == expected && sharder.stats().entries == 0 && sharder.stats().writes == 0,
          "failed registration lost data or changed counters");
    fs::remove(directory); fs::create_directory(directory);
    sharder.register_parameter("p", owner, true, &owner);
    owner->data<float>()[0] = 5.0f;
    sharder.mark_dirty("p");
    // Fail atomic replacement, preserving both the RAM copy and prior disk file.
    auto saved = directory / "saved.bin";
    fs::rename(directory / "0.bin", saved);
    fs::create_directory(directory / "0.bin");
    { std::ofstream blocker(directory / "0.bin" / "blocker"); blocker << 'x'; }
    throws([&] { sharder.offload_all(); });
    check(owner == expected && sharder.stats().resident_bytes == 8 && sharder.stats().writes == 1,
          "failed write lost residency or counted success");
    fs::remove_all(directory / "0.bin"); fs::rename(saved, directory / "0.bin");
    sharder.offload_all();
    check(!owner, "successful retry did not clear owner");
    fs::resize_file(directory / "0.bin", 3);
    const auto before = sharder.stats();
    throws([&] { sharder.require("p"); });
    check(!owner && sharder.stats().loads == before.loads && sharder.stats().resident_bytes == 0,
          "truncated read committed a partial tensor");
    fs::remove(directory / "0.bin");
    throws([&] { sharder.require("p"); });
    auto survivor = tensor({6.0f, 7.0f});
    sharder.register_parameter("survivor", survivor, true, &survivor);
    throws([&] { sharder.require("p"); });
    check(survivor && sharder.stats().resident_bytes == 8,
          "failed read evicted an unrelated resident");
    throws([&] { sharder.register_parameter("duplicate-owner", survivor, true, &survivor); });
    throws([&] { sharder.register_parameter("survivor", survivor); });
    // A clean resident can repair a missing/truncated backing file before release.
    fs::resize_file(directory / "1.bin", 1);
    sharder.evict("survivor");
    equal(sharder.require("survivor"), tensor({6.0f, 7.0f}));
    throws([&] { sharder.mark_dirty("p"); });
    throws([&] { sharder.mark_dirty("missing"); });
    throws([&] { sharder.require("missing"); });
    throws([&] { sharder.evict("missing"); });
    auto trainable = tensor({1.0f}); trainable->set_requires_grad(true);
    throws([&] { sharder.register_parameter("trainable", trainable); });
    throws([&] { sharder.register_parameter("half", empty({1}, kFloat16)); });
    throws([&] { sharder.register_parameter("big", zeros({3})); });
    throws([&] { sharder.register_parameter("null", nullptr); });
    throws([&] { ParameterSharder invalid({root.string(), 8, true}); });
    throws([&] { ParameterSharder invalid({root.string(), 0, false}); });
}
} // namespace

int main() {
    try {
        TemporaryDirectory directory;
        roundtrip(directory.path);
        failures(directory.path);
        std::cout << "[PASS] parameter offload primitives\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
