#include "tokenizer.h"
#include <sstream>
#include <algorithm>
#include <regex>
#include <fstream>
#include <iostream>

namespace ops {

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

    std::cout << "GPT2Tokenizer: Initialized " << vocab_.size() << " vocabulary tokens" << std::endl;
}

std::vector<std::string> GPT2Tokenizer::basic_tokenize(const std::string& text) {

    std::vector<std::string> tokens;
    std::string current_token;

    for (char c : text) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            if (!std::isspace(c)) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            current_token += std::tolower(c);
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

        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        if (vocab_.find(word) != vocab_.end()) {
            tokens.push_back(vocab_[word]);
        } else {

            size_t hash = std::hash<std::string>{}(word);
            int token_id = (hash % (vocab_size_ - 2000)) + 2000;
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

}
