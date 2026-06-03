#include "tokenizer.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_eq(int actual, int expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + " (actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected) + ")");
    }
}

void require_eq(const std::string& actual,
                const std::string& expected,
                const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + " (actual=\"" + actual +
            "\", expected=\"" + expected + "\")");
    }
}

fs::path make_case_dir(const std::string& name) {
    const fs::path root =
        fs::temp_directory_path() / ("mft_tokenizer_factory_" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

void test_config_inference() {
    const fs::path qwen = make_case_dir("qwen");
    write_file(qwen / "config.json", R"({"model_type":"qwen2","vocab_size":151936})");
    require(ops::TokenizerFactory::infer_model_type(qwen.string()) == "qwen2",
            "qwen model_type inference failed");

    const fs::path gemma = make_case_dir("gemma");
    write_file(gemma / "config.json", R"({"model_type":"gemma3_text"})");
    require(ops::TokenizerFactory::infer_model_type(gemma.string()) == "gemma",
            "gemma model_type inference failed");

    const fs::path gpt2 = make_case_dir("gpt2");
    write_file(gpt2 / "config.json", R"({"model_type":"gpt2"})");
    require(ops::TokenizerFactory::infer_model_type(gpt2.string()) == "gpt2",
            "gpt2 model_type inference failed");
}

void test_explicit_override_and_errors() {
    ops::TokenizerLoadOptions opts;

    const fs::path assets_without_config = make_case_dir("assets_without_config");
    write_file(assets_without_config / "vocab.json", R"({"Hello":0,"<|endoftext|>":50256})");
    write_file(assets_without_config / "merges.txt", "#version: 0.2\n");

    bool threw = false;
    try {
        (void)ops::TokenizerFactory::infer_model_type(assets_without_config.string());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("assets alone") != std::string::npos;
    }
    require(threw, "tokenizer assets without config.json should be ambiguous");

    opts.model_type = "gpt2";
    auto tokenizer = ops::TokenizerFactory::from_pretrained(assets_without_config.string(), opts);
    require(tokenizer != nullptr,
            "explicit gpt2 override adapter was not created");
    require_eq(tokenizer->decode(tokenizer->encode("Hello")), "Hello",
               "explicit gpt2 override encode/decode failed");
    opts.model_type.clear();

    const fs::path unknown = make_case_dir("unknown");
    threw = false;
    try {
        (void)ops::TokenizerFactory::infer_model_type(unknown.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "unknown tokenizer directory did not throw");

    const fs::path tokenizer_json_only = make_case_dir("tokenizer_json_only");
    write_file(tokenizer_json_only / "tokenizer.json", "{}");
    threw = false;
    try {
        (void)ops::TokenizerFactory::infer_model_type(tokenizer_json_only.string());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("assets alone") != std::string::npos;
    }
    require(threw, "tokenizer.json-only directory should be ambiguous");
}

void write_minimal_gpt2_assets(const fs::path& root) {
    write_file(root / "config.json", R"({"model_type":"gpt2"})");
    write_file(root / "vocab.json", R"({"Hello":0,"<|endoftext|>":50256})");
    write_file(root / "merges.txt", "#version: 0.2\n");
    write_file(root / "tokenizer.json", R"({"model":{"type":"BPE"}})");
}

void write_minimal_qwen_assets(const fs::path& root) {
    write_file(root / "config.json", R"({"model_type":"qwen2","vocab_size":11})");
    write_file(root / "vocab.json", R"({"Hello":0})");
    write_file(root / "merges.txt", "#version: 0.2\n");
    write_file(root / "tokenizer.json",
               R"({"added_tokens":[{"id":10,"content":"<|endoftext|>","special":true}]})");
}

void write_minimal_gemma_assets(const fs::path& root) {
    write_file(root / "config.json", R"({"model_type":"gemma3_text"})");
    write_file(root / "tokenizer.json",
               R"({"model":{"vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"Hello":4},"merges":[]}})");
}

void test_factory_loads_minimal_real_assets() {
    {
        const fs::path root = make_case_dir("load_gpt2");
        write_minimal_gpt2_assets(root);

        auto tokenizer = ops::TokenizerFactory::from_pretrained(root.string());
        require_eq(tokenizer->get_vocab_size(), 50257, "gpt2 vocab size mismatch");
        require_eq(tokenizer->get_eos_token(), 50256, "gpt2 eos mismatch");

        const auto ids = tokenizer->encode("Hello");
        require_eq(static_cast<int>(ids.size()), 1, "gpt2 encode length mismatch");
        require_eq(ids[0], 0, "gpt2 Hello token mismatch");
        require_eq(tokenizer->decode(ids), "Hello", "gpt2 decode mismatch");
    }

    {
        const fs::path root = make_case_dir("load_qwen");
        write_minimal_qwen_assets(root);

        auto tokenizer = ops::TokenizerFactory::from_pretrained(root.string());
        require_eq(tokenizer->get_vocab_size(), 11, "qwen vocab size mismatch");
        require_eq(tokenizer->get_eos_token(), 10, "qwen eos mismatch");
        require_eq(tokenizer->get_pad_token(), 10, "qwen pad mismatch");

        const auto ids = tokenizer->encode("Hello");
        require_eq(static_cast<int>(ids.size()), 1, "qwen encode length mismatch");
        require_eq(ids[0], 0, "qwen Hello token mismatch");
        require_eq(tokenizer->decode(ids), "Hello", "qwen decode mismatch");

        const auto encoded = tokenizer->encode_with_attention("Hello", 3);
        require_eq(static_cast<int>(encoded.input_ids.size()), 3, "qwen padded id length mismatch");
        require_eq(encoded.input_ids[0], 0, "qwen padded first token mismatch");
        require_eq(encoded.input_ids[1], 10, "qwen padded pad token mismatch");
        require_eq(encoded.attention_mask[0], 1, "qwen first mask mismatch");
        require_eq(encoded.attention_mask[1], 0, "qwen pad mask mismatch");
    }

    {
        const fs::path root = make_case_dir("load_gemma");
        write_minimal_gemma_assets(root);

        auto tokenizer = ops::TokenizerFactory::from_pretrained(root.string());
        require_eq(tokenizer->get_vocab_size(), 5, "gemma vocab size mismatch");
        require_eq(tokenizer->get_bos_token(), 2, "gemma bos mismatch");
        require_eq(tokenizer->get_pad_token(), 0, "gemma pad mismatch");

        const auto ids = tokenizer->encode("Hello");
        require_eq(static_cast<int>(ids.size()), 2, "gemma encode length mismatch");
        require_eq(ids[0], 2, "gemma bos token mismatch");
        require_eq(ids[1], 4, "gemma Hello token mismatch");
        require_eq(tokenizer->decode(ids), "Hello", "gemma decode mismatch");

        const auto encoded = tokenizer->encode_with_attention("Hello", 4);
        require_eq(encoded.input_ids[2], 0, "gemma pad token mismatch");
        require_eq(encoded.attention_mask[2], 0, "gemma pad mask mismatch");
    }
}

}  // namespace

int main() {
    test_config_inference();
    test_explicit_override_and_errors();
    test_factory_loads_minimal_real_assets();
    return 0;
}
