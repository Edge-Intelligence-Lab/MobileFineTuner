#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ops {

// Hash function for std::pair<std::string, std::string>
struct hash_pair {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer {
public:
    Tokenizer() = default;
    virtual ~Tokenizer() = default;

    virtual std::vector<int> encode(const std::string& text) = 0;

    virtual std::string decode(const std::vector<int>& tokens) = 0;

    virtual int get_vocab_size() const = 0;

    virtual int get_eos_token() const = 0;
    virtual int get_bos_token() const = 0;
    virtual int get_pad_token() const = 0;
    virtual int get_unk_token() const = 0;
};

class GPT2Tokenizer : public Tokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    std::unordered_map<std::pair<std::string, std::string>, int, hash_pair> bpe_merges_;

    int vocab_size_;
    int eos_token_ = 50256;
    int bos_token_ = 50256;
    int pad_token_ = 50256;
    int unk_token_ = 50256;

    std::vector<std::string> basic_tokenize(const std::string& text);
    std::vector<std::string> wordpiece_tokenize(const std::string& word);
    std::vector<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);

public:
    GPT2Tokenizer(int vocab_size = 50257);
    ~GPT2Tokenizer() override = default;

    void load_vocab(const std::string& vocab_file);
    void load_merges(const std::string& merges_file);

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

    int get_vocab_size() const override { return vocab_size_; }
    int get_eos_token() const override { return eos_token_; }
    int get_bos_token() const override { return bos_token_; }
    int get_pad_token() const override { return pad_token_; }
    int get_unk_token() const override { return unk_token_; }

    void init_simple_vocab();
};

class SimpleTokenizer : public Tokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    int vocab_size_;

public:
    SimpleTokenizer(int vocab_size = 50257);
    ~SimpleTokenizer() override = default;

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

    int get_vocab_size() const override { return vocab_size_; }
    int get_eos_token() const override { return 50256; }
    int get_bos_token() const override { return 50256; }
    int get_pad_token() const override { return 50256; }
    int get_unk_token() const override { return 50256; }
};

}
