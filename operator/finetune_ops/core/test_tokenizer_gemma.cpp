#include "tokenizer_gemma.h"

#include <iostream>
#include <stdexcept>

using namespace ops;

void expect_ids(const std::vector<int>& actual,
                const std::vector<int>& expected,
                const std::string& label) {
    if (actual != expected) {
        std::cerr << "[FAIL] " << label << std::endl;
        std::cerr << "  Expected: ";
        for (auto id : expected) std::cerr << id << " ";
        std::cerr << "\n  Actual: ";
        for (auto id : actual) std::cerr << id << " ";
        std::cerr << std::endl;
        throw std::runtime_error("Tokenizer mismatch: " + label);
    } else {
        std::cout << "[PASS] " << label << std::endl;
    }
}

void test_basic_alignment(GemmaTokenizer& tokenizer) {
    std::cout << "\n[Test] SentencePiece alignment..." << std::endl;

    auto ids = tokenizer.encode("Hello, world!\n", false, 0, false);
    expect_ids(ids, {9259, 236764, 1902, 236888, 107}, "Hello, world!\\n");

    ids = tokenizer.encode(u8"\u4eca\u5929\u5929\u6c14\u4e0d\u9519🙂", false, 0, false);
    expect_ids(ids, {22601, 122362, 56033, 240731}, "sample-cn-1");

    ids = tokenizer.encode("  leading space", false, 0, false);
    expect_ids(ids, {138, 26016, 2557}, "leading space");
}

void test_special_tokens(GemmaTokenizer& tokenizer) {
    std::cout << "\n[Test] Special tokens..." << std::endl;

    auto ids = tokenizer.encode("Hello, world!", true, 0, false);
    if (ids.empty() || ids.front() != tokenizer.get_bos_token_id()) {
        throw std::runtime_error("Missing BOS token when add_special_tokens=true");
    }
    if (!ids.empty() && ids.back() == tokenizer.get_eos_token_id()) {
        std::cout << "  (Note) Gemma does not add EOS automatically" << std::endl;
    }

    auto decoded = tokenizer.decode(ids, true);
    if (decoded != "Hello, world!") {
        throw std::runtime_error("Decode mismatch after stripping specials");
    }

    std::cout << "[PASS] Special token handling" << std::endl;
}

void test_byte_fallback(GemmaTokenizer& tokenizer) {
    std::cout << "\n[Test] Byte fallback..." << std::endl;
    std::string binary("\x01\x02", 2);
    auto ids = tokenizer.encode(binary, false, 0, false);
    expect_ids(ids, {249732, 241247}, "byte fallback ids");

    auto decoded = tokenizer.decode(ids, false);
    if (decoded != binary) {
        throw std::runtime_error("Byte fallback decode mismatch");
    }
    std::cout << "[PASS] Byte fallback roundtrip" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_gemma_dir>" << std::endl;
        return 1;
    }

    try {
        auto config = GemmaTokenizerConfig::from_pretrained(argv[1]);
        GemmaTokenizer tokenizer(config);
        tokenizer.load();

        test_basic_alignment(tokenizer);
        test_special_tokens(tokenizer);
        test_byte_fallback(tokenizer);

        std::cout << "\nAll Gemma tokenizer tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

