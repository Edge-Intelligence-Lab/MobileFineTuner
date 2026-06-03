#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct GoldenCase {
    std::string model_key;
    std::string model_type;
    std::string model_dir;
    std::string case_name;
    std::string text;
    std::vector<int> expected_ids;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    throw std::runtime_error("invalid JSON unicode escape");
}

uint32_t parse_u16_escape(const std::string& line, size_t hex_pos) {
    require(hex_pos + 4 <= line.size(), "short JSON unicode escape");
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value = (value << 4) | static_cast<uint32_t>(hex_value(line[hex_pos + i]));
    }
    return value;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        throw std::runtime_error("invalid JSON unicode codepoint");
    }
}

std::string parse_json_string_at(const std::string& line, size_t quote_pos) {
    require(quote_pos < line.size() && line[quote_pos] == '"', "invalid JSON string start");
    std::string out;
    for (size_t i = quote_pos + 1; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            require(i + 1 < line.size(), "unterminated JSON escape");
            char escaped = line[++i];
            switch (escaped) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_u16_escape(line, i + 1);
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        require(i + 2 < line.size() && line[i + 1] == '\\' && line[i + 2] == 'u',
                                "high surrogate without low surrogate");
                        uint32_t low = parse_u16_escape(line, i + 3);
                        require(low >= 0xDC00 && low <= 0xDFFF, "invalid low surrogate");
                        i += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        throw std::runtime_error("low surrogate without high surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    throw std::runtime_error("unsupported JSON escape in fixture");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

size_t value_start_for_key(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key in fixture: " + key);
    }
    const size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("missing colon for key in fixture: " + key);
    }
    size_t pos = colon + 1;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    return pos;
}

std::string json_string_field(const std::string& line, const std::string& key) {
    const size_t pos = value_start_for_key(line, key);
    return parse_json_string_at(line, pos);
}

std::vector<int> json_int_array_field(const std::string& line, const std::string& key) {
    size_t pos = value_start_for_key(line, key);
    require(pos < line.size() && line[pos] == '[', "expected array for key: " + key);
    ++pos;

    std::vector<int> values;
    while (pos < line.size()) {
        while (pos < line.size() &&
               (std::isspace(static_cast<unsigned char>(line[pos])) || line[pos] == ',')) {
            ++pos;
        }
        if (pos < line.size() && line[pos] == ']') {
            return values;
        }

        const size_t start = pos;
        if (pos < line.size() && line[pos] == '-') {
            ++pos;
        }
        while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        require(start != pos, "expected integer in array for key: " + key);
        values.push_back(std::stoi(line.substr(start, pos - start)));
    }

    throw std::runtime_error("unterminated array for key: " + key);
}

GoldenCase parse_case(const std::string& line) {
    GoldenCase item;
    item.model_key = json_string_field(line, "model_key");
    item.model_type = json_string_field(line, "model_type");
    item.model_dir = json_string_field(line, "model_dir");
    item.case_name = json_string_field(line, "case_name");
    item.text = json_string_field(line, "text");
    item.expected_ids = json_int_array_field(line, "expected_ids");
    require(!item.model_dir.empty(), "golden fixture model_dir is empty");
    return item;
}

std::string preview_ids(const std::vector<int>& ids, size_t max_count = 16) {
    std::string out = "[";
    const size_t count = std::min(ids.size(), max_count);
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(ids[i]);
    }
    if (ids.size() > max_count) {
        out += ",...";
    }
    out += "]";
    return out;
}

void compare_case(const GoldenCase& item, const std::vector<int>& actual_ids) {
    if (actual_ids == item.expected_ids) {
        return;
    }

    size_t first_mismatch = 0;
    const size_t common = std::min(actual_ids.size(), item.expected_ids.size());
    while (first_mismatch < common && actual_ids[first_mismatch] == item.expected_ids[first_mismatch]) {
        ++first_mismatch;
    }

    std::string message =
        "HF tokenizer golden mismatch for " + item.model_key + "/" + item.case_name +
        ": expected_len=" + std::to_string(item.expected_ids.size()) +
        ", actual_len=" + std::to_string(actual_ids.size()) +
        ", first_mismatch=" + std::to_string(first_mismatch) +
        ", expected=" + preview_ids(item.expected_ids) +
        ", actual=" + preview_ids(actual_ids);
    throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    const char* fixture_env = std::getenv("MFT_TOKENIZER_GOLDEN_JSONL");
    std::string fixture_path;
    if (argc > 1) {
        fixture_path = argv[1];
    } else if (fixture_env != nullptr && fixture_env[0] != '\0') {
        fixture_path = fixture_env;
    } else {
        std::cout << "[SKIP] MFT_TOKENIZER_GOLDEN_JSONL is not set; "
                  << "generate fixtures with scripts/generate_tokenizer_hf_golden_fixtures.py\n";
        return 0;
    }

    std::ifstream f(fixture_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open tokenizer golden fixture: " + fixture_path);
    }

    std::unordered_map<std::string, std::unique_ptr<ops::Tokenizer>> tokenizers;
    int checked = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        GoldenCase item = parse_case(line);
        auto it = tokenizers.find(item.model_dir);
        if (it == tokenizers.end()) {
            ops::TokenizerLoadOptions options;
            options.model_type = item.model_type;
            auto tokenizer = ops::TokenizerFactory::from_pretrained(item.model_dir, options);
            it = tokenizers.emplace(item.model_dir, std::move(tokenizer)).first;
        }

        const auto actual_ids = it->second->encode(item.text);
        compare_case(item, actual_ids);
        ++checked;
    }

    require(checked > 0, "tokenizer golden fixture contained no cases");
    std::cout << "HF tokenizer golden alignment passed for " << checked << " cases\n";
    return 0;
}
