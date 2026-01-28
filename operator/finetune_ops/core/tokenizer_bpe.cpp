/**
 * @file tokenizer_bpe.cpp
 * @brief GPT-2 Byte-Level BPE Tokenizer implementation (aligned with HuggingFace)
 *
 * Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
 */

#include "tokenizer_bpe.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <regex>
#include <cctype>
#include <climits>
#include <set>
#include <iterator>

// Lightweight JSON parser (supports \uXXXX escape sequences)
namespace simple_json {
    static std::string unescape_json_string(const std::string& s) {
        std::string result;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char next = s[i+1];
                if (next == 'n') { result += '\n'; i++; }
                else if (next == 't') { result += '\t'; i++; }
                else if (next == 'r') { result += '\r'; i++; }
                else if (next == '\\') { result += '\\'; i++; }
                else if (next == '"') { result += '"'; i++; }
                else if (next == 'u' && i + 5 < s.size()) {
                    // \uXXXX
                    std::string hex = s.substr(i+2, 4);
                    int codepoint = std::stoi(hex, nullptr, 16);
                    
                    // UTF-8 encoding
                    if (codepoint < 0x80) {
                        result += static_cast<char>(codepoint);
                    } else if (codepoint < 0x800) {
                        result += static_cast<char>(0xC0 | (codepoint >> 6));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else {
                        result += static_cast<char>(0xE0 | (codepoint >> 12));
                        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    }
                    i += 5;
                } else {
                    result += s[i];
                }
            } else {
                result += s[i];
            }
        }
        return result;
    }
    
    static std::unordered_map<std::string, int> parse_vocab_json(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + filepath);
        }
        
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();
        
        std::unordered_map<std::string, int> result;
        
        // Find all "key": value pairs
        std::regex pattern(R"#("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))#");
        auto words_begin = std::sregex_iterator(content.begin(), content.end(), pattern);
        auto words_end = std::sregex_iterator();
        
        for (auto it = words_begin; it != words_end; ++it) {
            std::string key_escaped = (*it)[1].str();
            std::string key = unescape_json_string(key_escaped);
            int value = std::stoi((*it)[2].str());
            result[key] = value;
        }
        
        return result;
    }

    struct AddedTokenInfo {
        std::string content;
        int id;
        bool special;
    };

    // Parse the added_tokens list in tokenizer.json
    static std::vector<AddedTokenInfo> parse_added_tokens(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + filepath);
        }

        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();

        std::vector<AddedTokenInfo> tokens;

        // Slice the added_tokens array only to avoid heavyweight regex on large files
        size_t start = content.find("\"added_tokens\"");
        if (start == std::string::npos) return tokens;
        start = content.find('[', start);
        if (start == std::string::npos) return tokens;
        size_t close = content.find(']', start);
        if (close == std::string::npos) return tokens;
        std::string slice = content.substr(start, close - start);

        std::regex item_re(
            R"REGEX("id"\s*:\s*(\d+)[^}]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)"[^}]*?"special"\s*:\s*(true|false))REGEX",
            std::regex::ECMAScript);

        auto begin = std::sregex_iterator(slice.begin(), slice.end(), item_re);
        auto end_it = std::sregex_iterator();
        for (auto it = begin; it != end_it; ++it) {
            int id = std::stoi((*it)[1].str());
            std::string token_raw = (*it)[2].str();
            bool special = (*it)[3].str() == "true" || (*it)[3].str() == "True";

            tokens.push_back({unescape_json_string(token_raw), id, special});
        }
        return tokens;
    }

    // Read vocab_size from config.json (return -1 if missing)
    static int parse_vocab_size(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            return -1;
        }

        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();

        std::regex pattern(R"("vocab_size"\s*:\s*(\d+))");
        std::smatch m;
        if (std::regex_search(content, m, pattern)) {
            try {
                return std::stoi(m[1].str());
            } catch (...) {
                return -1;
            }
        }
        return -1;
    }
}

namespace {

struct Utf8Char {
    uint32_t cp;
    size_t len;
};

// Decode a single UTF-8 character (tolerant: treat invalid sequences as single bytes)
static Utf8Char decode_utf8(const std::string& text, size_t index) {
    unsigned char c = static_cast<unsigned char>(text[index]);
    if ((c & 0x80) == 0) {
        return {c, 1};
    } else if ((c & 0xE0) == 0xC0 && index + 1 < text.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
        return {cp, 2};
    } else if ((c & 0xF0) == 0xE0 && index + 2 < text.size()) {
        uint32_t cp = ((c & 0x0F) << 12) |
                      ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(text[index + 2]) & 0x3F);
        return {cp, 3};
    } else if ((c & 0xF8) == 0xF0 && index + 3 < text.size()) {
        uint32_t cp = ((c & 0x07) << 18) |
                      ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 12) |
                      ((static_cast<unsigned char>(text[index + 2]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(text[index + 3]) & 0x3F);
        return {cp, 4};
    }
    return {c, 1};
}

static bool is_newline_cp(uint32_t cp) {
    return cp == 0x0A || cp == 0x0D;
}

static bool is_whitespace_cp(uint32_t cp) {
    if (cp < 128) {
        return std::isspace(static_cast<unsigned char>(cp)) != 0;
    }
    switch (cp) {
        case 0x00A0: case 0x1680: case 0x2000: case 0x2001: case 0x2002:
        case 0x2003: case 0x2004: case 0x2005: case 0x2006: case 0x2007:
        case 0x2008: case 0x2009: case 0x200A: case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return false;
    }
}

static bool is_number_cp(uint32_t cp) {
    if (cp < 128) {
        return std::isdigit(static_cast<unsigned char>(cp)) != 0;
    }
    if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9)) return true; // Arabic
    if (cp >= 0x0966 && cp <= 0x096F) return true; // Devanagari
    if (cp >= 0xFF10 && cp <= 0xFF19) return true; // Fullwidth
    return false;
}

static bool is_letter_cp(uint32_t cp) {
    if (cp < 128) {
        return std::isalpha(static_cast<unsigned char>(cp)) != 0;
    }
    if ((cp >= 0x00C0 && cp <= 0x02AF) ||     // Latin, IPA
        (cp >= 0x0370 && cp <= 0x03FF) ||     // Greek
        (cp >= 0x0400 && cp <= 0x052F) ||     // Cyrillic
        (cp >= 0x0530 && cp <= 0x058F) ||     // Armenian
        (cp >= 0x0590 && cp <= 0x05FF) ||     // Hebrew
        (cp >= 0x0600 && cp <= 0x06FF) ||     // Arabic
        (cp >= 0x0900 && cp <= 0x097F) ||     // Devanagari
        (cp >= 0x0E00 && cp <= 0x0E7F) ||     // Thai
        (cp >= 0x3040 && cp <= 0x30FF) ||     // Hiragana/Katakana
        (cp >= 0x3100 && cp <= 0x312F) ||     // Bopomofo
        (cp >= 0xAC00 && cp <= 0xD7AF) ||     // Hangul
        (cp >= 0x3400 && cp <= 0x4DBF) ||     // CJK Ext A
        (cp >= 0x4E00 && cp <= 0x9FFF) ||     // CJK
        (cp >= 0xF900 && cp <= 0xFAFF)) {     // CJK compatibility
        return true;
    }
    return false;
}

} // anonymous namespace

namespace ops {

// ============================================================================
// Construction and loading
// ============================================================================

GPT2BPETokenizer::GPT2BPETokenizer(const BPEConfig& config)
    : config_(config), vocab_size_(0) {
    build_byte_encoder();
}

void GPT2BPETokenizer::load() {
    load_vocab();
    load_merges();
    load_special_tokens();
    std::cout << "[GPT2BPETokenizer] Loaded: vocab_size=" << vocab_size_ 
              << ", merges=" << bpe_ranks_.size() << std::endl;
}

// ============================================================================
// Byte→Unicode mapping (exactly aligned with HuggingFace)
// ============================================================================

void GPT2BPETokenizer::build_byte_encoder() {
    // Identical to HF transformers bytes_to_unicode()
    std::vector<int> bs;
    std::vector<int> cs;
    
    // Ranges that can be used directly
    auto push_range = [&](int start, int end) {
        for (int i = start; i <= end; ++i) {
            bs.push_back(i);
            cs.push_back(i);
        }
    };
    
    push_range(int('!'), int('~'));   // 33-126
    push_range(161, 172);             // ¡-¬
    push_range(174, 255);             // ®-ÿ
    
    // Map remaining bytes to 256+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    
    // Build lookup tables
    byte_encoder_.clear();
    byte_decoder_.clear();
    
    for (size_t i = 0; i < 256; ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        int codepoint = cs[i];
        
        // UTF-8 encode the codepoint
        std::string utf8_char;
        if (codepoint < 0x80) {
            utf8_char = std::string(1, static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            utf8_char += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            utf8_char += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            utf8_char += static_cast<char>(0xF0 | (codepoint >> 18));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        
        byte_encoder_[byte_val] = utf8_char;
        byte_decoder_[utf8_char] = byte_val;
    }
}

std::string GPT2BPETokenizer::bytes_to_unicode(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 3);
    
    for (unsigned char byte : text) {
        result += byte_encoder_[byte];
    }
    
    return result;
}

std::string GPT2BPETokenizer::unicode_to_bytes(const std::string& unicode_text) {
    std::string result;
    result.reserve(unicode_text.size());
    
    // Decode as UTF-8 and reverse map each character back to its byte
    for (size_t i = 0; i < unicode_text.size(); ) {
        unsigned char c = unicode_text[i];
        int char_len = 1;
        
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        std::string utf8_char = unicode_text.substr(i, char_len);
        auto it = byte_decoder_.find(utf8_char);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        }
        
        i += char_len;
    }
    
    return result;
}

// ============================================================================
// Load vocab/merges/special_tokens
// ============================================================================

void GPT2BPETokenizer::load_vocab() {
    auto parsed = simple_json::parse_vocab_json(config_.vocab_path);
    
    for (auto& [token, id] : parsed) {
        vocab_[token] = id;
        id_to_token_[id] = token;
    }
    
    vocab_size_ = vocab_.size();
}

void GPT2BPETokenizer::load_merges() {
    std::ifstream f(config_.merges_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open merges.txt: " + config_.merges_path);
    }
    
    std::string line;
    int rank = 0;
    
    // Skip first line (#version: 0.2)
    std::getline(f, line);
    
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string a, b;
        if (iss >> a >> b) {
            std::string pair_key = a + " " + b;
            bpe_ranks_[pair_key] = rank++;
        }
    }
}

void GPT2BPETokenizer::load_special_tokens() {
    special_tokens_["<|endoftext|>"] = config_.eos_token_id;
}

// ============================================================================
// BPE core logic (full implementation)
// ============================================================================

std::vector<std::string> GPT2BPETokenizer::split_to_words(const std::string& text) {
    // GPT-2 regex after byte→unicode mapping:
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // Manual implementation without \p classes:
    // - English contractions: 's|'t|'re|'ve|'m|'ll|'d
    // - Letters/numbers with optional leading space
    // - Punctuation with optional leading space
    // - Consecutive whitespace
    std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d|)"
        R"( ?[a-zA-Z]+|)"
        R"( ?[0-9]+|)"
        R"( ?[^\s\w]+|)"
        R"(\s+(?!\S)|\s+)"
    );
    
    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    
    for (auto it = words_begin; it != words_end; ++it) {
        std::string match = it->str();
        if (!match.empty()) {
            words.push_back(match);
        }
    }
    
    return words;
}

std::pair<int, int> GPT2BPETokenizer::get_best_pair(const std::vector<std::string>& word) {
    if (word.size() < 2) return {-1, INT_MAX};
    
    int best_rank = INT_MAX;
    int best_i = -1;
    
    for (size_t i = 0; i < word.size() - 1; ++i) {
        std::string pair_key = word[i] + " " + word[i+1];
        auto it = bpe_ranks_.find(pair_key);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
            best_rank = it->second;
            best_i = static_cast<int>(i);
        }
    }
    
    return {best_i, best_rank};
}

std::vector<std::string> GPT2BPETokenizer::bpe(const std::string& token) {
    if (token.empty()) return {};
    
    // If the entire token exists in the vocab, return directly
    if (vocab_.find(token) != vocab_.end()) {
        return {token};
    }
    
    // Split into UTF-8 characters
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); ) {
        unsigned char c = token[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        
        if (i + len > token.size()) len = token.size() - i;
        word.push_back(token.substr(i, len));
        i += len;
    }
    
    // BPE merge loop
    while (word.size() > 1) {
        auto [best_i, best_rank] = get_best_pair(word);
        if (best_i == -1 || best_rank == INT_MAX) {
            break;
        }
        
        // Merge
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
            if (static_cast<int>(i) == best_i) {
                new_word.push_back(word[i] + word[i+1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
    }
    
    return word;
}

// ============================================================================
// Encode / Decode
// ============================================================================

std::vector<int> GPT2BPETokenizer::encode(const std::string& text,
                                          bool add_special_tokens,
                                          int max_length,
                                          bool truncation) {
    std::vector<int> ids;
    
    // 1. bytes → unicode mapping
    std::string unicode_text = bytes_to_unicode(text);
    
    // 2. Pre-tokenize
    auto words = split_to_words(unicode_text);
    
    // 3. Apply BPE to each word
    for (const auto& word : words) {
        auto bpe_tokens = bpe(word);
        
        for (const auto& bpe_token : bpe_tokens) {
            auto it = vocab_.find(bpe_token);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                // Byte-level should not hit OOV unless the vocab is corrupted
                std::cerr << "[ERROR] Token not in vocab: \"" << bpe_token << "\" (len=" << bpe_token.size() << ")" << std::endl;
            }
        }
    }
    
    // 4. Add special tokens
    if (add_special_tokens) {
        ids.push_back(config_.eos_token_id);
    }
    
    // 5. Truncate
    if (truncation && max_length > 0 && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
    }
    
    return ids;
}

std::string GPT2BPETokenizer::decode(const std::vector<int>& ids,
                                     bool skip_special_tokens) {
    std::string unicode_text;
    
    for (int id : ids) {
        // Skip special tokens
        if (skip_special_tokens && 
            (id == config_.eos_token_id || 
             id == config_.bos_token_id || 
             id == config_.pad_token_id)) {
            continue;
        }
        
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            unicode_text += it->second;
        }
    }
    
    // Unicode → bytes
    std::string result = unicode_to_bytes(unicode_text);
    return result;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
GPT2BPETokenizer::batch_encode(const std::vector<std::string>& texts,
                               int max_length,
                               const std::string& padding,
                               bool truncation) {
    std::vector<std::vector<int>> all_ids;
    std::vector<std::vector<int>> all_masks;
    
    for (const auto& text : texts) {
        auto ids = encode(text, false, max_length, truncation);
        all_ids.push_back(ids);
    }
    
    int target_len = 0;
    if (padding == "max_length" && max_length > 0) {
        target_len = max_length;
    } else if (padding == "longest") {
        for (const auto& ids : all_ids) {
            target_len = std::max(target_len, static_cast<int>(ids.size()));
        }
    }
    
    for (auto& ids : all_ids) {
        int orig_len = ids.size();
        std::vector<int> mask(orig_len, 1);
        
        if (target_len > 0 && orig_len < target_len) {
            int pad_count = target_len - orig_len;
            ids.insert(ids.end(), pad_count, config_.pad_token_id);
            mask.insert(mask.end(), pad_count, 0);
        }
        
        all_masks.push_back(mask);
    }
    
    return {all_ids, all_masks};
}

std::string GPT2BPETokenizer::get_token_string(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return "";
}

// ============================================================================
// QwenBPETokenizer
// ============================================================================

QwenBPETokenizer::QwenBPETokenizer(const QwenTokenizerConfig& cfg)
    : config_(cfg) {
    build_byte_encoder();
    eos_token_id_ = cfg.eos_token_id;
    bos_token_id_ = cfg.bos_token_id;
    pad_token_id_ = cfg.pad_token_id;
}

void QwenBPETokenizer::build_byte_encoder() {
    std::vector<int> bs;
    std::vector<int> cs;

    auto push_range = [&](int start, int end) {
        for (int i = start; i <= end; ++i) {
            bs.push_back(i);
            cs.push_back(i);
        }
    };

    push_range(int('!'), int('~'));
    push_range(161, 172);
    push_range(174, 255);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    byte_encoder_.clear();
    byte_decoder_.clear();
    for (size_t i = 0; i < 256; ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        int codepoint = cs[i];

        std::string utf8_char;
        if (codepoint < 0x80) {
            utf8_char = std::string(1, static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            utf8_char += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            utf8_char += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            utf8_char += static_cast<char>(0xF0 | (codepoint >> 18));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        }

        byte_encoder_[byte_val] = utf8_char;
        byte_decoder_[utf8_char] = byte_val;
    }
}

std::string QwenBPETokenizer::bytes_to_unicode(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 3);
    for (unsigned char byte : text) {
        result += byte_encoder_[byte];
    }
    return result;
}

std::string QwenBPETokenizer::unicode_to_bytes(const std::string& unicode_text) {
    std::string result;
    result.reserve(unicode_text.size());

    for (size_t i = 0; i < unicode_text.size();) {
        unsigned char c = unicode_text[i];
        int char_len = 1;

        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }

        std::string utf8_char = unicode_text.substr(i, char_len);
        auto it = byte_decoder_.find(utf8_char);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        }

        i += char_len;
    }

    return result;
}

void QwenBPETokenizer::load_vocab() {
    auto parsed = simple_json::parse_vocab_json(config_.vocab_path);
    for (auto& [token, id] : parsed) {
        vocab_[token] = id;
        id_to_token_[id] = token;
    }
    vocab_size_ = static_cast<int>(vocab_.size());
}

void QwenBPETokenizer::load_merges() {
    std::ifstream f(config_.merges_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open merges.txt: " + config_.merges_path);
    }

    std::string line;
    int rank = 0;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string a, b;
        if (iss >> a >> b) {
            bpe_ranks_[a + " " + b] = rank++;
        }
    }
}

void QwenBPETokenizer::load_added_tokens() {
    added_tokens_.clear();
    id_to_added_.clear();
    added_tokens_sorted_.clear();
    added_special_flags_.clear();

    if (config_.tokenizer_json_path.empty()) {
        return;
    }

    try {
        auto toks = simple_json::parse_added_tokens(config_.tokenizer_json_path);
        for (const auto& t : toks) {
            added_tokens_[t.content] = t.id;
            id_to_added_[t.id] = t.content;
            added_special_flags_[t.content] = t.special;
            if (t.content == "<|endoftext|>") {
                eos_token_id_ = t.id;
                pad_token_id_ = t.id;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Failed to parse added_tokens: " << e.what() << std::endl;
    }

    added_tokens_sorted_.reserve(added_tokens_.size());
    for (const auto& kv : added_tokens_) {
        added_tokens_sorted_.push_back(kv.first);
    }
    std::sort(added_tokens_sorted_.begin(), added_tokens_sorted_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    int max_added_id = -1;
    for (const auto& kv : added_tokens_) {
        max_added_id = std::max(max_added_id, kv.second);
    }
    if (max_added_id >= 0) {
        vocab_size_ = std::max(vocab_size_, max_added_id + 1);
    }
}

void QwenBPETokenizer::load_vocab_size_override() {
    int override_val = config_.vocab_size_override;
    if (override_val <= 0 && !config_.config_json_path.empty()) {
        override_val = simple_json::parse_vocab_size(config_.config_json_path);
    }
    if (override_val > 0) {
        vocab_size_ = std::max(vocab_size_, override_val);
    }
}

void QwenBPETokenizer::load() {
    load_vocab();
    load_merges();
    load_added_tokens();
    load_vocab_size_override();
    std::cout << "[QwenBPETokenizer] Loaded: vocab_size=" << vocab_size_
              << ", merges=" << bpe_ranks_.size()
              << ", added_tokens=" << added_tokens_.size() << std::endl;
}

std::pair<int, int> QwenBPETokenizer::get_best_pair(const std::vector<std::string>& word) {
    if (word.size() < 2) return {-1, INT_MAX};
    int best_rank = INT_MAX;
    int best_i = -1;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        std::string key = word[i] + " " + word[i + 1];
        auto it = bpe_ranks_.find(key);
        if (it != bpe_ranks_.end() && it->second < best_rank) {
            best_rank = it->second;
            best_i = static_cast<int>(i);
        }
    }
    return {best_i, best_rank};
}

std::vector<std::string> QwenBPETokenizer::bpe(const std::string& token) {
    if (token.empty()) return {};
    if (vocab_.find(token) != vocab_.end()) {
        return {token};
    }

    std::vector<std::string> word;
    for (size_t i = 0; i < token.size();) {
        unsigned char c = token[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > token.size()) len = token.size() - i;
        word.push_back(token.substr(i, len));
        i += len;
    }

    while (word.size() > 1) {
        auto [best_i, best_rank] = get_best_pair(word);
        if (best_i == -1 || best_rank == INT_MAX) break;

        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size();) {
            if (static_cast<int>(i) == best_i) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word.swap(new_word);
    }
    return word;
}

std::vector<QwenBPETokenizer::Segment> QwenBPETokenizer::split_by_added_tokens(const std::string& text) {
    std::vector<Segment> segments;
    if (added_tokens_.empty()) {
        segments.push_back({text, false});
        return segments;
    }

    std::string buffer;
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& tok : added_tokens_sorted_) {
            if (tok.empty()) continue;
            if (i + tok.size() <= text.size() && text.compare(i, tok.size(), tok) == 0) {
                if (!buffer.empty()) {
                    segments.push_back({buffer, false});
                    buffer.clear();
                }
                segments.push_back({tok, true});
                i += tok.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;
        buffer.push_back(text[i]);
        i++;
    }
    if (!buffer.empty()) {
        segments.push_back({buffer, false});
    }
    return segments;
}

std::vector<std::string> QwenBPETokenizer::split_to_words(const std::string& text) {
    // Same pre-tokenization regex as GPT-2 (ASCII approximation)
    std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d|)"
        R"( ?[a-zA-Z]+|)"
        R"( ?[0-9]+|)"
        R"( ?[^\s\w]+|)"
        R"(\s+(?!\S)|\s+)"
    );
    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    for (auto it = words_begin; it != words_end; ++it) {
        std::string match = it->str();
        if (!match.empty()) words.push_back(match);
    }
    return words;
}

std::vector<int> QwenBPETokenizer::encode(const std::string& text,
                                          bool add_special_tokens,
                                          int max_length,
                                          bool truncation) {
    std::vector<int> ids;

    auto segments = split_by_added_tokens(text);
    for (const auto& seg : segments) {
        if (seg.is_added) {
            auto it = added_tokens_.find(seg.text);
            if (it != added_tokens_.end()) {
                ids.push_back(it->second);
            }
            continue;
        }

        // GPT-2 style: regex tokenize raw text, then byte→unicode + BPE per segment
        auto words = split_to_words(seg.text);
        for (const auto& word_raw : words) {
            auto unicode_word = bytes_to_unicode(word_raw);
            auto bpe_tokens = bpe(unicode_word);
            for (const auto& t : bpe_tokens) {
                auto it = vocab_.find(t);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                } else {
                    std::cerr << "[WARN] Token not in vocab: \"" << t << "\"\n";
                }
            }
        }
    }

    if (add_special_tokens && eos_token_id_ >= 0) {
        ids.push_back(eos_token_id_);
    }

    if (truncation && max_length > 0 && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
    }
    return ids;
}

std::string QwenBPETokenizer::decode(const std::vector<int>& ids,
                                     bool skip_special_tokens) {
    std::string unicode_text;

    for (int id : ids) {
        if (skip_special_tokens) {
            if ((id == eos_token_id_) || (id == pad_token_id_) || (id == bos_token_id_)) continue;
            auto add_it = id_to_added_.find(id);
            if (add_it != id_to_added_.end()) {
                if (added_special_flags_[add_it->second]) continue;
            }
        }

        auto add_it = id_to_added_.find(id);
        if (add_it != id_to_added_.end()) {
            unicode_text += add_it->second;
            continue;
        }

        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            unicode_text += it->second;
        }
    }

    return unicode_to_bytes(unicode_text);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
QwenBPETokenizer::batch_encode(const std::vector<std::string>& texts,
                               int max_length,
                               const std::string& padding,
                               bool truncation) {
    std::vector<std::vector<int>> all_ids;
    std::vector<std::vector<int>> all_masks;

    for (const auto& text : texts) {
        all_ids.push_back(encode(text, false, max_length, truncation));
    }

    int target_len = 0;
    if (padding == "max_length" && max_length > 0) {
        target_len = max_length;
    } else if (padding == "longest") {
        for (const auto& ids : all_ids) {
            target_len = std::max(target_len, static_cast<int>(ids.size()));
        }
    }

    for (auto& ids : all_ids) {
        int orig_len = static_cast<int>(ids.size());
        std::vector<int> mask(orig_len, 1);
        if (target_len > 0 && orig_len < target_len) {
            int pad_count = target_len - orig_len;
            ids.insert(ids.end(), pad_count, pad_token_id_);
            mask.insert(mask.end(), pad_count, 0);
        }
        all_masks.push_back(mask);
    }

    return {all_ids, all_masks};
}

std::string QwenBPETokenizer::get_token_string(int token_id) const {
    auto it_added = id_to_added_.find(token_id);
    if (it_added != id_to_added_.end()) return it_added->second;
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) return it->second;
    return "";
}

}  // namespace ops
