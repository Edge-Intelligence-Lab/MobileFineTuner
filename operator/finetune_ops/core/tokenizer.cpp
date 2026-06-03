#include "tokenizer.h"
#include "tokenizer_bpe.h"
#include "tokenizer_gemma.h"

#include <sstream>
#include <algorithm>
#include <regex>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cctype>
#include <utility>

namespace {

namespace fs = std::filesystem;

std::string read_file(const fs::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string extract_json_string(const std::string& content, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
        return match[1].str();
    }
    return "";
}

bool path_exists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

}  // namespace

namespace ops {

EncodedInput Tokenizer::encode_with_attention(const std::string& text,
                                              int max_length,
                                              bool truncation) {
    EncodedInput result;
    result.input_ids = encode(text);

    if (truncation && max_length > 0 && static_cast<int>(result.input_ids.size()) > max_length) {
        result.input_ids.resize(max_length);
    }

    result.attention_mask.assign(result.input_ids.size(), 1);

    if (max_length > 0 && static_cast<int>(result.input_ids.size()) < max_length) {
        int pad = get_pad_token();
        if (pad < 0) {
            pad = get_eos_token();
        }
        if (pad < 0) {
            pad = 0;
        }
        const size_t old_size = result.input_ids.size();
        result.input_ids.resize(max_length, pad);
        result.attention_mask.resize(max_length, 0);
        std::fill(result.attention_mask.begin(), result.attention_mask.begin() + old_size, 1);
    }

    return result;
}

TokenizerSpecialTokens Tokenizer::special_tokens() const {
    return {
        get_bos_token(),
        get_eos_token(),
        get_pad_token(),
        get_unk_token(),
    };
}

GPT2Tokenizer::GPT2Tokenizer(int vocab_size) : vocab_size_(vocab_size) {
    init_simple_vocab();
}

void GPT2Tokenizer::init_simple_vocab() {

    for (int i = 0; i < 256; ++i) {
        std::string byte_char(1, static_cast<char>(i));
        vocab_[byte_char] = i;
        reverse_vocab_[i] = byte_char;
    }

    std::vector<std::string> common_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "up", "about", "into", "through", "during", "before", "after", "above", "below",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "can", "shall",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "this", "that", "these", "those", "here", "there", "where", "when", "why", "how", "what", "which", "who",
        "machine", "learning", "artificial", "intelligence", "computer", "science", "technology", "data",
        "algorithm", "neural", "network", "deep", "model", "training", "research", "development",
        "language", "processing", "natural", "text", "analysis", "prediction", "classification",
        "history", "began", "first", "started", "early", "development", "field", "study", "work"
    };

    int token_id = 256;
    for (const auto& word : common_words) {
        if (token_id < vocab_size_ - 1000) {
            vocab_[word] = token_id;
            reverse_vocab_[token_id] = word;
            token_id++;
        }
    }

    for (int i = token_id; i < vocab_size_; ++i) {
        std::string token = "tok_" + std::to_string(i);
        vocab_[token] = i;
        reverse_vocab_[i] = token;
    }

}

std::vector<std::string> GPT2Tokenizer::basic_tokenize(const std::string& text) {

    std::vector<std::string> tokens;
    std::string current_token;

    for (char c : text) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isspace(uc) || std::ispunct(uc)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            if (!std::isspace(uc)) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            current_token += static_cast<char>(std::tolower(uc));
        }
    }

    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    return tokens;
}

std::vector<int> GPT2Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;

    auto words = basic_tokenize(text);

    for (const auto& word : words) {
        if (vocab_.find(word) != vocab_.end()) {
            tokens.push_back(vocab_[word]);
        } else {

            for (char c : word) {
                std::string char_str(1, c);
                if (vocab_.find(char_str) != vocab_.end()) {
                    tokens.push_back(vocab_[char_str]);
                } else {
                    tokens.push_back(unk_token_);
                }
            }
        }
    }

    return tokens;
}

std::vector<std::string> GPT2Tokenizer::wordpiece_tokenize(const std::string& word) {
    return {word};
}

std::vector<std::pair<std::string, std::string>> GPT2Tokenizer::get_pairs(
    const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) {
        return pairs;
    }
    pairs.reserve(word.size() - 1);
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs.emplace_back(word[i], word[i + 1]);
    }
    return pairs;
}

void GPT2Tokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream f(vocab_file);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_file);
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    const std::string content = buffer.str();

    vocab_.clear();
    reverse_vocab_.clear();
    int max_id = -1;

    const std::regex pattern(R"#("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))#");
    for (auto it = std::sregex_iterator(content.begin(), content.end(), pattern);
         it != std::sregex_iterator();
         ++it) {
        const std::string token = (*it)[1].str();
        const int id = std::stoi((*it)[2].str());
        vocab_[token] = id;
        reverse_vocab_[id] = token;
        max_id = std::max(max_id, id);
    }

    if (max_id >= 0) {
        vocab_size_ = max_id + 1;
    }
}

void GPT2Tokenizer::load_merges(const std::string& merges_file) {
    std::ifstream f(merges_file);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open merges file: " + merges_file);
    }

    bpe_merges_.clear();
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string left;
        std::string right;
        if (iss >> left >> right) {
            bpe_merges_[{left, right}] = rank++;
        }
    }
}

std::string GPT2Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;

    for (int token : tokens) {
        if (reverse_vocab_.find(token) != reverse_vocab_.end()) {
            std::string token_str = reverse_vocab_[token];

            if (token_str.length() == 1 && token < 256) {
                result += token_str;
            } else {

                if (!result.empty() && result.back() != ' ') {
                    result += " ";
                }
                result += token_str;
            }
        }
    }

    return result;
}

SimpleTokenizer::SimpleTokenizer(int vocab_size) : vocab_size_(vocab_size) {

    for (int i = 0; i < vocab_size; ++i) {
        std::string token = "token_" + std::to_string(i);
        vocab_[token] = i;
        reverse_vocab_[i] = token;
    }

    std::vector<std::string> words = {
        "the", "a", "and", "is", "in", "of", "to", "for", "with", "on", "at", "by", "from",
        "machine", "learning", "artificial", "intelligence", "computer", "science",
        "history", "began", "technology", "data", "model", "training", "research"
    };

    for (size_t i = 0; i < words.size() && i < 1000; ++i) {
        vocab_[words[i]] = 1000 + i;
        reverse_vocab_[1000 + i] = words[i];
    }
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {

        word.erase(
            std::remove_if(word.begin(), word.end(), [](unsigned char c) {
                return std::ispunct(c) != 0;
            }),
            word.end());
        std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (vocab_.find(word) != vocab_.end()) {
            tokens.push_back(vocab_[word]);
        } else {

            size_t hash = std::hash<std::string>{}(word);
            const int offset = vocab_size_ > 2000 ? 2000 : 0;
            const int bucket_size = std::max(1, vocab_size_ - offset);
            int token_id = offset + static_cast<int>(hash % bucket_size);
            if (token_id >= vocab_size_) {
                token_id = 0;
            }
            tokens.push_back(token_id);
        }
    }

    if (tokens.empty()) {
        tokens.push_back(get_eos_token());
    }

    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int>& tokens) {
    std::string result;

    for (int token : tokens) {
        if (reverse_vocab_.find(token) != reverse_vocab_.end()) {
            if (!result.empty()) result += " ";
            result += reverse_vocab_[token];
        } else {
            if (!result.empty()) result += " ";
            result += "unk_" + std::to_string(token);
        }
    }

    return result;
}

namespace {

class GPT2BPEAdapter final : public Tokenizer {
public:
    explicit GPT2BPEAdapter(const BPEConfig& config)
        : tokenizer_(config) {
        tokenizer_.load();
    }

    std::vector<int> encode(const std::string& text) override {
        return tokenizer_.encode(text);
    }

    std::string decode(const std::vector<int>& tokens) override {
        return tokenizer_.decode(tokens);
    }

    int get_vocab_size() const override { return tokenizer_.get_vocab_size(); }
    int get_eos_token() const override { return tokenizer_.get_eos_token_id(); }
    int get_bos_token() const override { return tokenizer_.get_bos_token_id(); }
    int get_pad_token() const override { return tokenizer_.get_pad_token_id(); }
    int get_unk_token() const override { return tokenizer_.get_unk_token_id(); }

private:
    GPT2BPETokenizer tokenizer_;
};

class QwenBPEAdapter final : public Tokenizer {
public:
    explicit QwenBPEAdapter(const QwenTokenizerConfig& config)
        : tokenizer_(config) {
        tokenizer_.load();
    }

    std::vector<int> encode(const std::string& text) override {
        return tokenizer_.encode(text);
    }

    std::string decode(const std::vector<int>& tokens) override {
        return tokenizer_.decode(tokens);
    }

    int get_vocab_size() const override { return tokenizer_.get_vocab_size(); }
    int get_eos_token() const override { return tokenizer_.get_eos_token_id(); }
    int get_bos_token() const override { return tokenizer_.get_bos_token_id(); }
    int get_pad_token() const override { return tokenizer_.get_pad_token_id(); }
    int get_unk_token() const override { return -1; }

private:
    QwenBPETokenizer tokenizer_;
};

class GemmaTokenizerAdapter final : public Tokenizer {
public:
    explicit GemmaTokenizerAdapter(GemmaTokenizerConfig config)
        : tokenizer_(std::move(config)) {
        tokenizer_.load();
    }

    std::vector<int> encode(const std::string& text) override {
        return tokenizer_.encode(text);
    }

    std::string decode(const std::vector<int>& tokens) override {
        return tokenizer_.decode(tokens);
    }

    int get_vocab_size() const override { return tokenizer_.get_vocab_size(); }
    int get_eos_token() const override { return tokenizer_.get_eos_token_id(); }
    int get_bos_token() const override { return tokenizer_.get_bos_token_id(); }
    int get_pad_token() const override { return tokenizer_.get_pad_token_id(); }
    int get_unk_token() const override { return tokenizer_.get_unk_token_id(); }

private:
    GemmaTokenizer tokenizer_;
};

}  // namespace

std::string TokenizerFactory::infer_model_type(const std::string& model_dir) {
    const fs::path root(model_dir);
    const fs::path config_path = root / "config.json";

    if (path_exists(config_path)) {
        const std::string config = read_file(config_path);
        const std::string model_type = lowercase(extract_json_string(config, "model_type"));
        if (model_type.find("qwen") != std::string::npos) {
            return "qwen2";
        }
        if (model_type.find("gemma") != std::string::npos) {
            return "gemma";
        }
        if (model_type.find("gpt2") != std::string::npos ||
            model_type.find("gpt_2") != std::string::npos ||
            model_type == "gpt") {
            return "gpt2";
        }
    }

    if (path_exists(root / "tokenizer.model")) {
        throw std::runtime_error(
            "SentencePiece tokenizer.model assets are not supported by MobileFineTuner yet; "
            "provide config.json with model_type and a tokenizer.json-backed Gemma tokenizer, "
            "or add a SentencePiece adapter.");
    }

    if (path_exists(root / "tokenizer.json") ||
        path_exists(root / "vocab.json") ||
        path_exists(root / "merges.txt")) {
        throw std::runtime_error(
            "Cannot infer tokenizer type from tokenizer assets alone. Provide config.json with "
            "model_type or pass TokenizerLoadOptions::model_type explicitly.");
    }

    throw std::runtime_error("Cannot infer tokenizer type for model directory: " + model_dir);
}

std::unique_ptr<Tokenizer> TokenizerFactory::from_pretrained(
    const std::string& model_dir,
    const TokenizerLoadOptions& options) {
    const std::string model_type = lowercase(
        options.model_type.empty() ? infer_model_type(model_dir) : options.model_type);

    if (model_type.find("qwen") != std::string::npos) {
        return std::make_unique<QwenBPEAdapter>(
            QwenTokenizerConfig::from_pretrained(model_dir));
    }

    if (model_type.find("gemma") != std::string::npos) {
        return std::make_unique<GemmaTokenizerAdapter>(
            GemmaTokenizerConfig::from_pretrained(model_dir));
    }

    if (model_type.find("gpt2") != std::string::npos ||
        model_type.find("gpt_2") != std::string::npos ||
        model_type == "gpt") {
        return std::make_unique<GPT2BPEAdapter>(
            BPEConfig::from_pretrained(model_dir));
    }

    throw std::runtime_error("Unsupported tokenizer model_type: " + model_type);
}

}
