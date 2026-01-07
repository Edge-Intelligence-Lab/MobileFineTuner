#include "mmlu_runner.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>

#include "finetune_ops/core/ops.h"

namespace fs = std::filesystem;

namespace ops {

static inline std::string trim_copy(const std::string& s) {
    size_t l = 0, r = s.size();
    while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
    while (r > l && std::isspace(static_cast<unsigned char>(s[r-1]))) --r;
    return s.substr(l, r - l);
}

MMLURunner::MMLURunner(std::string mmlu_root, std::string split)
: mmlu_root_(std::move(mmlu_root)), split_(std::move(split)) {}

std::vector<std::string> MMLURunner::parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i+1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') {
                fields.emplace_back(std::move(cur));
                cur.clear();
            } else if (c == '"') {
                in_quotes = true;
            } else {
                cur.push_back(c);
            }
        }
    }
    fields.emplace_back(std::move(cur));
    return fields;
}

void MMLURunner::read_mmlu_csv(const std::string& path, std::vector<MCQItem>& out_items) {
    std::ifstream in(path);
    if (!in) return;

    std::string header;
    if (!std::getline(in, header)) return;
    auto cols = parse_csv_line(header);

    auto find_col = [&](const std::string& name)->int{
        for (size_t i = 0; i < cols.size(); ++i) {
            std::string c = trim_copy(cols[i]);
            std::transform(c.begin(), c.end(), c.begin(), ::tolower);
            if (c == name) return static_cast<int>(i);
        }
        return -1;
    };

    int idx_subject = find_col("subject");
    int idx_question = find_col("question");
    int idx_a = find_col("a");
    int idx_b = find_col("b");
    int idx_c = find_col("c");
    int idx_d = find_col("d");
    int idx_answer = find_col("answer");

    if (idx_question < 0 || idx_a < 0 || idx_b < 0 || idx_c < 0 || idx_d < 0 || idx_answer < 0) {
        // 简易容错
    }

    std::string line;
    while (std::getline(in, line)) {
        if (trim_copy(line).empty()) continue;
        auto f = parse_csv_line(line);
        if (static_cast<int>(f.size()) <= std::max({idx_subject, idx_question, idx_a, idx_b, idx_c, idx_d, idx_answer})) continue;
        MCQItem item;
        item.subject = (idx_subject >= 0 ? trim_copy(f[idx_subject]) : std::string("unknown"));
        item.question = trim_copy(f[idx_question]);
        item.A = trim_copy(f[idx_a]);
        item.B = trim_copy(f[idx_b]);
        item.C = trim_copy(f[idx_c]);
        item.D = trim_copy(f[idx_d]);
        std::string ans = trim_copy(f[idx_answer]);
        item.answer = ans.empty() ? 'A' : static_cast<char>(std::toupper(static_cast<unsigned char>(ans[0])));
        out_items.emplace_back(std::move(item));
    }
}

void MMLURunner::load() {
    subject_to_items_.clear();
    std::string split_dir = mmlu_root_ + "/" + split_;

    for (auto& p : fs::directory_iterator(split_dir)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension() != ".csv") continue;
        std::vector<MCQItem> items;
        read_mmlu_csv(p.path().string(), items);
        for (auto& x : items) {
            subject_to_items_[x.subject].push_back(std::move(x));
        }
    }
}

std::string MMLURunner::build_prompt(const MCQItem& x, const std::vector<MCQItem>* shots) {
    std::string prompt;
    auto append_one = [&](const MCQItem& q){
        prompt += "Question: " + q.question + "\n";
        prompt += "A. " + q.A + "\n";
        prompt += "B. " + q.B + "\n";
        prompt += "C. " + q.C + "\n";
        prompt += "D. " + q.D + "\n";
        prompt += "Answer: ";
    };

    if (shots && !shots->empty()) {
        for (const auto& s : *shots) {
            append_one(s);
            prompt += static_cast<char>(s.answer);
            prompt += "\n\n";
        }
    }
    append_one(x);
    return prompt;
}

char MMLURunner::predict_letter(const std::string& prompt, GPT2Model& model, GPT2BPETokenizer& tok) {
    // 1) Encode prompt without special tokens
    auto ids_vec = tok.encode(prompt, false, 0, false);
    if (ids_vec.empty()) ids_vec.push_back(0);

    std::vector<int32_t> ids(ids_vec.begin(), ids_vec.end());
    std::vector<float> attn(ids.size(), 1.0f);

    TensorPtr input_ids = std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(ids.size())},
        ids.data(), kInt32, kCPU);
    TensorPtr attention = std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(ids.size())},
        attn.data(), kFloat32, kCPU);

    // 2) Forward pass
    auto logits = model.forward(input_ids, attention);
    // 3) Extract logits for the last position [1, V]
    auto logits2d = flatten(logits, 0, 1);
    int64_t S = logits->shape()[1];
    int64_t V = logits2d->shape()[1];
    const float* all = logits2d->data<float>();
    std::vector<float> last_row(V);
    const float* src = all + (S - 1) * V;
    std::copy(src, src + V, last_row.begin());
    TensorPtr last_logits = std::make_shared<Tensor>(
        std::vector<int64_t>{1, V}, last_row.data(), kFloat32, kCPU);

    // 4) Compute log-softmax
    auto logp = log_softmax(last_logits, 1);
    const float* lp = logp->data<float>();
    
    // 5) Resolve option token ids
    auto idA_vec = tok.encode("A", false, 0, false);
    auto idB_vec = tok.encode("B", false, 0, false);
    auto idC_vec = tok.encode("C", false, 0, false);
    auto idD_vec = tok.encode("D", false, 0, false);
    int idA = idA_vec.empty() ? 0 : idA_vec[0];
    int idB = idB_vec.empty() ? 1 : idB_vec[0];
    int idC = idC_vec.empty() ? 2 : idC_vec[0];
    int idD = idD_vec.empty() ? 3 : idD_vec[0];

    auto get_lp = [&](int idx)->float { return (idx >=0 && idx < V) ? lp[idx] : -1e30f; };
    float sA = get_lp(idA);
    float sB = get_lp(idB);
    float sC = get_lp(idC);
    float sD = get_lp(idD);

    char pred = 'A';
    float best = sA;
    if (sB > best) { best = sB; pred = 'B'; }
    if (sC > best) { best = sC; pred = 'C'; }
    if (sD > best) { best = sD; pred = 'D'; }
    return pred;
}

MMLUResult MMLURunner::evaluate(GPT2Model& model, GPT2BPETokenizer& tok, int fewshot_k) {
    MMLUResult result;
    int total_correct = 0;
    int total_count = 0;

    result.per_subject.reserve(subject_to_items_.size());

    for (auto& kv : subject_to_items_) {
        const std::string& subj = kv.first;
        const auto& items = kv.second;
        if (items.empty()) continue;

        int correct = 0;
        int count = 0;

        std::vector<MCQItem> shots;
        if (fewshot_k > 0) {
            for (size_t i = 0; i < static_cast<size_t>(fewshot_k) && i < items.size(); ++i) shots.push_back(items[i]);
        }

        for (size_t i = 0; i < items.size(); ++i) {
            const auto& x = items[i];
            std::vector<MCQItem> shots_ex;
            if (fewshot_k > 0) {
                shots_ex.reserve(shots.size());
                for (const auto& s : shots) if (&s != &x) shots_ex.push_back(s);
            }
            auto prompt = build_prompt(x, fewshot_k > 0 ? &shots_ex : nullptr);
            char pred = predict_letter(prompt, model, tok);
            if (pred == x.answer) correct++;
            count++;
        }

        SubjectReport rep;
        rep.subject = subj;
        rep.correct = correct;
        rep.total = count;
        result.per_subject.push_back(rep);

        total_correct += correct;
        total_count += count;
    }

    // Sort subjects alphabetically for stable reporting
    std::sort(result.per_subject.begin(), result.per_subject.end(), [](const SubjectReport& a, const SubjectReport& b){
        return a.subject < b.subject;
    });

    float macro = 0.0f;
    for (const auto& r : result.per_subject) macro += r.accuracy();
    if (!result.per_subject.empty()) macro /= static_cast<float>(result.per_subject.size());
    float micro = (total_count > 0) ? static_cast<float>(total_correct) / static_cast<float>(total_count) : 0.0f;

    result.macro = macro;
    result.micro = micro;
    return result;
}

} // namespace ops


