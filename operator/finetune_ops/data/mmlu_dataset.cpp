/**
 * @file mmlu_dataset.cpp
 * @brief MMLU dataset loader implementation
 */

#include "mmlu_dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cstring>

namespace fs = std::filesystem;

namespace ops {

// -------------- MMLUSample --------------

std::string MMLUSample::to_prompt() const {
    std::ostringstream ss;
    ss << "Question: " << question << "\n";
    ss << "A. " << choices[0] << "\n";
    ss << "B. " << choices[1] << "\n";
    ss << "C. " << choices[2] << "\n";
    ss << "D. " << choices[3] << "\n";
    ss << "Answer:";
    return ss.str();
}

std::string MMLUSample::to_full_text() const {
    return to_prompt() + " " + std::string(1, answer);
}

// -------------- MMLUDataset --------------

MMLUDataset::MMLUDataset(const MMLUConfig& cfg, QwenBPETokenizer* tok)
    : cfg_(cfg), tok_(tok), rng_(cfg.seed) {
}

void MMLUDataset::load_train() {
    samples_.clear();
    load_from_dir("dev");  // MMLU dev split used for few-shot or training
    
    // Build sequential indices
    order_.resize(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) {
        order_[i] = i;
    }
    cursor_ = 0;
    
    std::cout << "[MMLU] Loaded " << samples_.size() << " training samples from dev/\n";
}

void MMLUDataset::load_test() {
    samples_.clear();
    load_from_dir("test");
    
    order_.resize(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) {
        order_[i] = i;
    }
    cursor_ = 0;
    
    std::cout << "[MMLU] Loaded " << samples_.size() << " test samples\n";
}

void MMLUDataset::load_val() {
    samples_.clear();
    load_from_dir("val");
    
    order_.resize(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) {
        order_[i] = i;
    }
    cursor_ = 0;
    
    std::cout << "[MMLU] Loaded " << samples_.size() << " validation samples\n";
}

void MMLUDataset::load_from_dir(const std::string& subdir) {
    std::string dir_path = cfg_.data_dir + "/" + subdir;
    
    // Optional subject filter set
    std::unordered_set<std::string> target_subjects;
    if (!cfg_.subjects.empty()) {
        for (const auto& s : cfg_.subjects) {
            target_subjects.insert(s);
        }
    }
    
    // Traverse all CSV files under the directory
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        if (filename.size() < 5 || filename.substr(filename.size() - 4) != ".csv") {
            continue;
        }
        
        // Derive subject name: strip suffix like _dev.csv / _test.csv / _val.csv
        std::string subject = filename;
        size_t pos = subject.rfind('_');
        if (pos != std::string::npos) {
            subject = subject.substr(0, pos);
        }
        
        // If a subject list is provided, ensure it is included
        if (!target_subjects.empty() && target_subjects.find(subject) == target_subjects.end()) {
            continue;
        }
        
        // Read the CSV file
        std::ifstream f(entry.path());
        if (!f.is_open()) {
            std::cerr << "[MMLU] Warning: cannot open " << entry.path() << std::endl;
            continue;
        }
        
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            
            try {
                MMLUSample sample = parse_csv_line(line, subject);
                samples_.push_back(std::move(sample));
            } catch (const std::exception& e) {
                // Skip malformed rows
                continue;
            }
        }
    }
}

MMLUSample MMLUDataset::parse_csv_line(const std::string& line, const std::string& subject) {
    // MMLU CSV format: question,A,B,C,D,answer
    // Note: question may contain commas, so quotes must be handled
    
    MMLUSample sample;
    sample.subject = subject;
    
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);  // last field
    
    if (fields.size() < 6) {
        throw std::runtime_error("Invalid CSV line: not enough fields");
    }
    
    sample.question = fields[0];
    sample.choices[0] = fields[1];
    sample.choices[1] = fields[2];
    sample.choices[2] = fields[3];
    sample.choices[3] = fields[4];
    
    // Answer is the last field and should be one of A/B/C/D
    std::string ans = fields[5];
    // Trim whitespace
    while (!ans.empty() && (ans.back() == ' ' || ans.back() == '\r' || ans.back() == '\n')) {
        ans.pop_back();
    }
    while (!ans.empty() && ans.front() == ' ') {
        ans = ans.substr(1);
    }
    
    if (ans.size() == 1 && ans[0] >= 'A' && ans[0] <= 'D') {
        sample.answer = ans[0];
    } else {
        throw std::runtime_error("Invalid answer: " + ans);
    }
    
    return sample;
}

size_t MMLUDataset::num_batches() const {
    return (samples_.size() + cfg_.batch_size - 1) / cfg_.batch_size;
}

void MMLUDataset::shuffle() {
    std::shuffle(order_.begin(), order_.end(), rng_);
    cursor_ = 0;
}

void MMLUDataset::init_token_cache() const {
    if (cache_initialized_) return;
    
    // Pre-encode common prefixes
    cached_prompt_prefix_ = tok_->encode("Question: ");
    cached_answer_prefix_ = tok_->encode("\nAnswer:");
    
    cache_initialized_ = true;
}

void MMLUDataset::encode_sample(const MMLUSample& sample,
                                std::vector<int32_t>& input_ids,
                                std::vector<int32_t>& labels) const {
    init_token_cache();
    
    // Encode full text
    std::string full_text = sample.to_full_text();
    std::vector<int32_t> full_ids = tok_->encode(full_text);
    
    // Encode prompt (without answer)
    std::string prompt = sample.to_prompt();
    std::vector<int32_t> prompt_ids = tok_->encode(prompt);
    
    // Truncate to seq_len
    if (full_ids.size() > static_cast<size_t>(cfg_.seq_len)) {
        full_ids.resize(cfg_.seq_len);
        // If prompt is longer than seq_len, trim it as well
        if (prompt_ids.size() > static_cast<size_t>(cfg_.seq_len)) {
            prompt_ids.resize(cfg_.seq_len);
        }
    }
    
    input_ids = full_ids;
    
    // Create labels: only answer tokens keep labels, others set to -100
    // Causal LM: labels[i] = input_ids[i+1], but we only compute loss on answer positions
    labels.resize(input_ids.size(), -100);
    
    // Locate answer tokens
    // Answers follow the prompt, typically like " A" or " B"
    // full_ids is prompt_ids plus answer tokens
    
    size_t prompt_len = prompt_ids.size();
    
    // For causal LM we predict input_ids[i+1], so:
    // labels[prompt_len - 1] = answer_token_id (predict the answer)
    // In practice answer tokens are at full_ids[prompt_len] or later.
    //
    // Simplified: place answer tokens on labels starting at the last prompt position
    if (prompt_len > 0 && prompt_len <= input_ids.size()) {
        // Locate answer tokens (in full_ids starting at prompt_len)
        for (size_t i = prompt_len; i < full_ids.size(); ++i) {
            // Set label: position i-1 predicts token at position i
            if (i - 1 < labels.size()) {
                labels[i - 1] = full_ids[i];
            }
        }
    }
}

MMLUBatch MMLUDataset::get_batch(size_t batch_idx) const {
    MMLUBatch batch;
    
    size_t start = batch_idx * cfg_.batch_size;
    size_t end = std::min(start + cfg_.batch_size, samples_.size());
    size_t actual_batch_size = end - start;
    
    if (actual_batch_size == 0) {
        batch.num_samples = 0;
        return batch;
    }
    
    // Allocate tensors
    batch.input_ids = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(actual_batch_size), cfg_.seq_len},
        kInt32, kCPU);
    batch.attention_mask = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(actual_batch_size), cfg_.seq_len},
        kFloat32, kCPU);
    batch.labels = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(actual_batch_size), cfg_.seq_len},
        kInt32, kCPU);
    
    int32_t* input_ptr = batch.input_ids->data<int32_t>();
    float* mask_ptr = batch.attention_mask->data<float>();
    int32_t* label_ptr = batch.labels->data<int32_t>();
    
    // Initialize to padding values
    std::memset(input_ptr, 0, actual_batch_size * cfg_.seq_len * sizeof(int32_t));
    std::memset(mask_ptr, 0, actual_batch_size * cfg_.seq_len * sizeof(float));
    for (size_t i = 0; i < actual_batch_size * cfg_.seq_len; ++i) {
        label_ptr[i] = -100;
    }
    
    // Fill each sample
    for (size_t b = 0; b < actual_batch_size; ++b) {
        size_t sample_idx = order_[start + b];
        const auto& sample = samples_[sample_idx];
        
        std::vector<int32_t> input_ids;
        std::vector<int32_t> labels;
        encode_sample(sample, input_ids, labels);
        
        size_t len = std::min(input_ids.size(), static_cast<size_t>(cfg_.seq_len));
        
        for (size_t i = 0; i < len; ++i) {
            input_ptr[b * cfg_.seq_len + i] = input_ids[i];
            mask_ptr[b * cfg_.seq_len + i] = 1.0f;
            label_ptr[b * cfg_.seq_len + i] = labels[i];
        }
    }
    
    batch.num_samples = actual_batch_size;
    return batch;
}

MMLUBatch MMLUDataset::next_batch(bool loop) {
    if (cursor_ >= num_batches()) {
        if (loop) {
            cursor_ = 0;
        } else {
            MMLUBatch empty;
            empty.num_samples = 0;
            return empty;
        }
    }
    
    return get_batch(cursor_++);
}

std::vector<std::string> MMLUDataset::get_subjects() const {
    std::unordered_set<std::string> unique_subjects;
    for (const auto& s : samples_) {
        unique_subjects.insert(s.subject);
    }
    return std::vector<std::string>(unique_subjects.begin(), unique_subjects.end());
}

void MMLUDataset::print_stats() const {
    std::cout << "\n========== MMLU Dataset Stats ==========\n";
    std::cout << "Total samples: " << samples_.size() << "\n";
    std::cout << "Batch size: " << cfg_.batch_size << "\n";
    std::cout << "Num batches: " << num_batches() << "\n";
    std::cout << "Seq length: " << cfg_.seq_len << "\n";
    
    // Count by subject
    std::unordered_map<std::string, int> subject_counts;
    for (const auto& s : samples_) {
        subject_counts[s.subject]++;
    }
    std::cout << "Subjects: " << subject_counts.size() << "\n";
    
    // Print top-5 subjects
    int count = 0;
    for (const auto& kv : subject_counts) {
        if (count++ >= 5) {
            std::cout << "  ... and " << (subject_counts.size() - 5) << " more\n";
            break;
        }
        std::cout << "  - " << kv.first << ": " << kv.second << " samples\n";
    }
    std::cout << "=========================================\n\n";
}

}  // namespace ops

