#include "tokenizer_gemma.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::string unescape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            switch (next) {
                case 'n': result.push_back('\n'); i++; break;
                case 't': result.push_back('\t'); i++; break;
                case 'r': result.push_back('\r'); i++; break;
                case '\\': result.push_back('\\'); i++; break;
                case '"': result.push_back('"'); i++; break;
                case 'u':
                    if (i + 5 < s.size()) {
                        std::string hex = s.substr(i + 2, 4);
                        int codepoint = std::stoi(hex, nullptr, 16);
                        if (codepoint < 0x80) {
                            result.push_back(static_cast<char>(codepoint));
                        } else if (codepoint < 0x800) {
                            result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
                            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
                        } else {
                            result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
                            result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
                        }
                        i += 5;
                    } else {
                        result.push_back(next);
                        i++;
                    }
                    break;
                default:
                    result.push_back(next);
                    i++;
                    break;
            }
        } else {
            result.push_back(c);
        }
    }
    return result;
}

std::string extract_json_block(const std::string& content,
                               size_t start_pos,
                               char open_char,
                               char close_char) {
    if (start_pos >= content.size() || content[start_pos] != open_char) {
        throw std::runtime_error("Invalid JSON block start");
    }

    int depth = 0;
    bool in_string = false;
    bool escape = false;
    size_t begin = start_pos;

    for (size_t i = start_pos; i < content.size(); ++i) {
        char c = content[i];

        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }

        if (c == open_char) {
            depth++;
        } else if (c == close_char) {
            depth--;
            if (depth == 0) {
                return content.substr(begin, i - begin + 1);
            }
        }
    }

    throw std::runtime_error("Unterminated JSON block while parsing tokenizer.json");
}

}  // namespace

namespace ops {

GemmaTokenizerConfig GemmaTokenizerConfig::from_pretrained(const std::string& model_dir) {
    GemmaTokenizerConfig cfg;
    cfg.tokenizer_json_path = model_dir + "/tokenizer.json";
    return cfg;
}

GemmaTokenizer::GemmaTokenizer(GemmaTokenizerConfig config)
    : config_(std::move(config)) {}

void GemmaTokenizer::load() {
    load_tokenizer_json();

    auto assign_id = [&](const std::string& token, int& target) {
        auto it = vocab_.find(token);
        if (it == vocab_.end()) {
            throw std::runtime_error("Token not found in vocab: " + token);
        }
        target = it->second;
    };

    assign_id(config_.bos_token, bos_token_id_);
    assign_id(config_.eos_token, eos_token_id_);
    assign_id(config_.pad_token, pad_token_id_);
    assign_id(config_.unk_token, unk_token_id_);
    vocab_size_ = static_cast<int>(vocab_.size());

    auto add_special = [&](const std::string& token) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            special_token_ids_.insert(it->second);
        }
    };

    add_special(config_.bos_token);
    add_special(config_.eos_token);
    add_special(config_.pad_token);
    add_special(config_.unk_token);
    for (const auto& token : config_.special_tokens) {
        add_special(token);
    }
}

std::vector<int> GemmaTokenizer::encode(const std::string& text,
                                        bool add_special_tokens,
                                        int max_length,
                                        bool truncation) const {
    std::vector<int> ids;

    if (add_special_tokens && config_.add_bos_token) {
        ids.push_back(bos_token_id_);
    }

    std::string normalized = normalize_text(text);
    auto tokens = bpe(normalized);
    for (const auto& token : tokens) {
        append_token_id(token, ids);
    }

    if (add_special_tokens && config_.add_eos_token) {
        ids.push_back(eos_token_id_);
    }

    if (truncation && max_length > 0 && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
    }

    return ids;
}

std::string GemmaTokenizer::decode(const std::vector<int>& ids,
                                   bool skip_special_tokens) const {
    std::string sentencepiece_text;

    for (int id : ids) {
        if (skip_special_tokens && special_token_ids_.count(id) > 0) {
            continue;
        }

        auto token_it = id_to_token_.find(id);
        if (token_it == id_to_token_.end()) {
            continue;
        }

        const std::string& token = token_it->second;
        if (is_byte_fallback_token(token)) {
            sentencepiece_text.push_back(decode_byte_token(token));
        } else {
            sentencepiece_text += token;
        }
    }

    return convert_sentencepiece_to_text(sentencepiece_text);
}

void GemmaTokenizer::load_tokenizer_json() {
    std::ifstream f(config_.tokenizer_json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open tokenizer json: " + config_.tokenizer_json_path);
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string content = buffer.str();

    size_t model_pos = content.find("\"model\"");
    if (model_pos == std::string::npos) {
        throw std::runtime_error("tokenizer.json missing \"model\" block");
    }

    size_t vocab_key = content.find("\"vocab\"", model_pos);
    if (vocab_key == std::string::npos) {
        throw std::runtime_error("tokenizer.json missing \"vocab\" block");
    }
    size_t vocab_start = content.find('{', vocab_key);
    if (vocab_start == std::string::npos) {
        throw std::runtime_error("Failed to locate vocab object in tokenizer.json");
    }

    std::string vocab_block = extract_json_block(content, vocab_start, '{', '}');
    parse_vocab_block(vocab_block);

    size_t merges_key = content.find("\"merges\"", vocab_key);
    if (merges_key == std::string::npos) {
        throw std::runtime_error("tokenizer.json missing \"merges\" array");
    }
    size_t merges_start = content.find('[', merges_key);
    if (merges_start == std::string::npos) {
        throw std::runtime_error("Failed to locate merges array in tokenizer.json");
    }

    std::string merges_block = extract_json_block(content, merges_start, '[', ']');
    parse_merges_block(merges_block);
}

void GemmaTokenizer::parse_vocab_block(const std::string& block) {
    vocab_.clear();
    id_to_token_.clear();

    std::regex pattern(R"#("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))#");
    auto begin = std::sregex_iterator(block.begin(), block.end(), pattern);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        std::string token = unescape_json_string((*it)[1].str());
        int id = std::stoi((*it)[2].str());
        vocab_[token] = id;
        id_to_token_[id] = token;
    }
}

void GemmaTokenizer::parse_merges_block(const std::string& block) {
    bpe_ranks_.clear();
    std::regex pattern(R"#(\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\])#");
    auto begin = std::sregex_iterator(block.begin(), block.end(), pattern);
    auto end = std::sregex_iterator();

    int rank = 0;
    for (auto it = begin; it != end; ++it, ++rank) {
        std::string first = unescape_json_string((*it)[1].str());
        std::string second = unescape_json_string((*it)[2].str());
        bpe_ranks_[{first, second}] = rank;
    }
}

std::string GemmaTokenizer::normalize_text(const std::string& text) const {
    std::string normalized;
    normalized.reserve(text.size() + text.size() / 2);

    for (unsigned char c : text) {
        if (c == ' ') {
            normalized += sentencepiece_space_;
        } else {
            normalized.push_back(static_cast<char>(c));
        }
    }
    return normalized;
}

std::vector<std::string> GemmaTokenizer::bpe(const std::string& token) const {
    if (token.empty()) {
        return {};
    }

    if (vocab_.find(token) != vocab_.end()) {
        return {token};
    }

    std::vector<std::string> word;
    for (size_t i = 0; i < token.size();) {
        unsigned char c = token[i];
        size_t len = 1;
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        if (i + len > token.size()) {
            len = token.size() - i;
        }
        word.emplace_back(token.substr(i, len));
        i += len;
    }

    while (word.size() > 1) {
        auto [best_index, best_rank] = get_best_pair(word);
        if (best_index < 0 || best_rank == std::numeric_limits<int>::max()) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (size_t i = 0; i < word.size();) {
            if (static_cast<int>(i) == best_index) {
                merged.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                merged.push_back(word[i]);
                i += 1;
            }
        }
        word.swap(merged);
    }

    return word;
}

std::pair<int, int> GemmaTokenizer::get_best_pair(const std::vector<std::string>& word) const {
    if (word.size() < 2) {
        return {-1, std::numeric_limits<int>::max()};
    }

    int best_index = -1;
    int best_rank = std::numeric_limits<int>::max();

    for (size_t i = 0; i + 1 < word.size(); ++i) {
        auto key = std::make_pair(word[i], word[i + 1]);
        auto it = bpe_ranks_.find(key);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
            best_rank = it->second;
            best_index = static_cast<int>(i);
        }
    }

    return {best_index, best_rank};
}

void GemmaTokenizer::append_token_id(const std::string& token, std::vector<int>& ids) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        ids.push_back(it->second);
    } else {
        append_byte_fallback(token, ids);
    }
}

void GemmaTokenizer::append_byte_fallback(const std::string& token, std::vector<int>& ids) const {
    for (unsigned char byte : token) {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "<0x%02X>", byte);
        auto fb = vocab_.find(buffer);
        if (fb != vocab_.end()) {
            ids.push_back(fb->second);
        } else {
            ids.push_back(unk_token_id_);
        }
    }
}

bool GemmaTokenizer::is_byte_fallback_token(const std::string& token) const {
    if (token.size() != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') {
        return false;
    }
    return std::isxdigit(static_cast<unsigned char>(token[3])) &&
           std::isxdigit(static_cast<unsigned char>(token[4]));
}

char GemmaTokenizer::decode_byte_token(const std::string& token) const {
    int value = std::stoi(token.substr(3, 2), nullptr, 16);
    return static_cast<char>(value);
}

std::string GemmaTokenizer::convert_sentencepiece_to_text(const std::string& text) const {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        if (c == 0xE2 && i + 2 < text.size()) {
            unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if (c1 == 0x96 && c2 == 0x81) {
                result.push_back(' ');
                i += 3;
                continue;
            }
        }
        result.push_back(static_cast<char>(c));
        i += 1;
    }

    return result;
}

}  // namespace ops

