#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/core/tokenizer_bpe.h"

namespace ops {

struct MCQItem {
    std::string subject;
    std::string question;
    std::string A, B, C, D;
    char answer; // 'A'|'B'|'C'|'D'
};

struct SubjectReport {
    std::string subject;
    int correct = 0;
    int total = 0;
    float accuracy() const { return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f; }
};

struct MMLUResult {
    std::vector<SubjectReport> per_subject;
    float macro = 0.0f;
    float micro = 0.0f;
};

class MMLURunner {
public:
    // mmlu_root points to data/mmlu/data; split is either "dev" or "test"
    MMLURunner(std::string mmlu_root, std::string split);

    // Scan the split directory, read all *.csv files and group rows by subject
    void load();

    // Run evaluation and return macro/micro accuracy along with per-subject breakdown
    MMLUResult evaluate(GPT2Model& model, GPT2BPETokenizer& tok, int fewshot_k = 0);

private:
    std::string mmlu_root_;
    std::string split_;
    std::unordered_map<std::string, std::vector<MCQItem>> subject_to_items_;

    static std::vector<std::string> parse_csv_line(const std::string& line);
    static void read_mmlu_csv(const std::string& path, std::vector<MCQItem>& out_items);
    static std::string build_prompt(const MCQItem& x, const std::vector<MCQItem>* shots = nullptr);
    static char predict_letter(const std::string& prompt, GPT2Model& model, GPT2BPETokenizer& tok);
};

} // namespace ops


