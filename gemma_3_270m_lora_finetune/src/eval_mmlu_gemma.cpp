#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <filesystem>

#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/tokenizer_gemma.h"
#include "finetune_ops/core/ops.h"

using namespace std;
using namespace ops;
namespace fs = std::filesystem;

struct Args {
    string mmlu_root = "data/mmlu/data";
    string split = "dev"; // dev|test
    string pretrained_dir = "pretrained";
    string lora_path;
    bool lora_merge = true; // placeholder flag if LoRA merge is added later
    int fewshot = 0;
    string out_file;
};

static void usage(const char* prog) {
    cerr << "Usage: " << prog <<
        " --mmlu_root PATH [--split dev|test] [--fewshot K] [--pretrained_dir PATH]" << endl;
    cerr << "  [--out FILE]" << endl;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get = [&](const string& key){ if (i+1>=argc || k!=key) { usage(argv[0]); exit(1);} return string(argv[++i]); };
        if (k == "--mmlu_root") a.mmlu_root = get("--mmlu_root");
        else if (k == "--split") a.split = get("--split");
        else if (k == "--fewshot") a.fewshot = stoi(get("--fewshot"));
        else if (k == "--pretrained_dir") a.pretrained_dir = get("--pretrained_dir");
        else if (k == "--out") a.out_file = get("--out");
        else if (k == "--help" || k == "-h") { usage(argv[0]); exit(0);} 
        else { cerr << "Unknown arg: " << k << endl; usage(argv[0]); exit(1); }
    }
    if (a.split != "dev" && a.split != "test") {
        cerr << "Invalid --split: " << a.split << ", must be dev|test" << endl;
        exit(1);
    }
    return a;
}

static inline string trim_copy(const string& s) {
    size_t l = 0, r = s.size();
    while (l < r && isspace(static_cast<unsigned char>(s[l]))) ++l;
    while (r > l && isspace(static_cast<unsigned char>(s[r-1]))) --r;
    return s.substr(l, r - l);
}

static vector<string> parse_csv_line(const string& line) {
    vector<string> fields;
    string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i+1] == '"') {
                    cur.push_back('"'); ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') { fields.emplace_back(move(cur)); cur.clear(); }
            else if (c == '"') { in_quotes = true; }
            else { cur.push_back(c); }
        }
    }
    fields.emplace_back(move(cur));
    return fields;
}

struct MCQItem {
    string subject;
    string question;
    string A, B, C, D;
    char answer;
};

static void read_mmlu_csv(const string& path, vector<MCQItem>& out_items) {
    ifstream in(path);
    if (!in) return;
    string header;
    if (!getline(in, header)) return;
    auto cols = parse_csv_line(header);
    auto find_col = [&](const string& name)->int{
        for (size_t i = 0; i < cols.size(); ++i) {
            string c = trim_copy(cols[i]);
            transform(c.begin(), c.end(), c.begin(), ::tolower);
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
    string line;
    while (getline(in, line)) {
        if (trim_copy(line).empty()) continue;
        auto f = parse_csv_line(line);
        if (static_cast<int>(f.size()) <= max({idx_subject, idx_question, idx_a, idx_b, idx_c, idx_d, idx_answer})) continue;
        MCQItem item;
        item.subject = (idx_subject >= 0 ? trim_copy(f[idx_subject]) : string("unknown"));
        item.question = trim_copy(f[idx_question]);
        item.A = trim_copy(f[idx_a]);
        item.B = trim_copy(f[idx_b]);
        item.C = trim_copy(f[idx_c]);
        item.D = trim_copy(f[idx_d]);
        string ans = trim_copy(f[idx_answer]);
        item.answer = ans.empty() ? 'A' : static_cast<char>(toupper(static_cast<unsigned char>(ans[0])));
        out_items.emplace_back(move(item));
    }
}

static string build_prompt(const MCQItem& q, const vector<MCQItem>* shots) {
    auto one = [](const MCQItem& x) {
        return string("Question: ") + x.question + "\n"
             + "A. " + x.A + "\n"
             + "B. " + x.B + "\n"
             + "C. " + x.C + "\n"
             + "D. " + x.D + "\n"
             + "Answer: ";
    };
    string prompt;
    if (shots && !shots->empty()) {
        for (const auto& s : *shots) {
            prompt += one(s);
            prompt += s.answer;
            prompt += "\n\n";
        }
    }
    prompt += one(q);
    return prompt;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);
    try {
        cout << "========== MMLU Evaluation (Gemma) ==========" << endl;
        cout << "mmlu_root     : " << args.mmlu_root << endl;
        cout << "split         : " << args.split << endl;
        cout << "fewshot       : " << args.fewshot << endl;
        cout << "pretrained_dir: " << args.pretrained_dir << endl;

        auto cfg = GemmaTextConfig::from_pretrained(args.pretrained_dir);
        GemmaModel model(cfg);
        SafeTensorsReader reader(args.pretrained_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) model.assign_weight(kv.first, kv.second);

        auto tok_cfg = GemmaTokenizerConfig::from_pretrained(args.pretrained_dir);
        GemmaTokenizer tok(tok_cfg);
        tok.load();

        unordered_map<string, vector<MCQItem>> subj2items;
        string split_dir = args.mmlu_root + "/" + args.split;
        for (auto& p : fs::directory_iterator(split_dir)) {
            if (!p.is_regular_file() || p.path().extension() != ".csv") continue;
            vector<MCQItem> items;
            read_mmlu_csv(p.path().string(), items);
            for (auto& it : items) subj2items[it.subject].push_back(move(it));
        }
        cout << "[Eval] Loaded " << subj2items.size() << " subjects" << endl;

        auto encode_ids = [&](const string& text)->vector<int32_t>{
            auto ids = tok.encode(text, false, 0, false);
            return vector<int32_t>(ids.begin(), ids.end());
        };

        auto last_letter = [&](const string& prompt)->char{
            auto ids32 = encode_ids(prompt);
            if (ids32.empty()) ids32.push_back(tok.get_pad_token_id() >= 0 ? tok.get_pad_token_id() : 0);
            vector<float> attn(ids32.size(), 1.0f);
            TensorPtr input_ids = make_shared<Tensor>(vector<int64_t>{1,(int64_t)ids32.size()}, ids32.data(), kInt32, kCPU);
            TensorPtr attention = make_shared<Tensor>(vector<int64_t>{1,(int64_t)ids32.size()}, attn.data(), kFloat32, kCPU);
            auto logits = model.forward(input_ids, attention); // [1,S,V]
            auto logits2d = flatten(logits, 0, 1);
            int64_t S = logits->shape()[1], V = logits2d->shape()[1];
            const float* all = logits2d->data<float>();
            vector<float> last_row(V);
            const float* src = all + (S - 1) * V;
            copy(src, src + V, last_row.begin());
            TensorPtr last_logits = make_shared<Tensor>(vector<int64_t>{1,V}, last_row.data(), kFloat32, kCPU);
            auto logp = log_softmax(last_logits, 1);
            const float* lp = logp->data<float>();
            // Leading space tends to be correct for SP-based models
            auto idA = tok.encode(" A", false, 0, false); int idxA = idA.empty()? tok.encode("A", false,0,false).front(): idA.back();
            auto idB = tok.encode(" B", false, 0, false); int idxB = idB.empty()? tok.encode("B", false,0,false).front(): idB.back();
            auto idC = tok.encode(" C", false, 0, false); int idxC = idC.empty()? tok.encode("C", false,0,false).front(): idC.back();
            auto idD = tok.encode(" D", false, 0, false); int idxD = idD.empty()? tok.encode("D", false,0,false).front(): idD.back();
            auto get_lp = [&](int idx)->float{ return (idx>=0 && idx < V) ? lp[idx] : -1e30f; };
            float sA = get_lp(idxA), sB = get_lp(idxB), sC = get_lp(idxC), sD = get_lp(idxD);
            char pred = 'A'; float best = sA;
            if (sB > best) { best = sB; pred = 'B'; }
            if (sC > best) { best = sC; pred = 'C'; }
            if (sD > best) { best = sD; pred = 'D'; }
            return pred;
        };

        struct Report { string subject; int correct=0; int total=0; };
        vector<Report> reports;
        int total_correct=0, total_count=0;
        for (auto& kv : subj2items) {
            const string& subj = kv.first;
            auto& items = kv.second;
            if (items.empty()) continue;
            int correct=0, count=0;
            vector<MCQItem> shots;
            if (args.fewshot > 0) {
                for (size_t i=0; i < (size_t)args.fewshot && i<items.size(); ++i) shots.push_back(items[i]);
            }
            for (size_t i=0; i<items.size(); ++i) {
                const auto& x = items[i];
                vector<MCQItem> shots_ex;
                if (args.fewshot > 0) {
                    shots_ex.reserve(shots.size());
                    for (const auto& s : shots) if (&s != &x) shots_ex.push_back(s);
                }
                auto prompt = build_prompt(x, args.fewshot > 0 ? &shots_ex : nullptr);
                char pred = last_letter(prompt);
                if (pred == x.answer) correct++;
                count++;
            }
            reports.push_back({subj, correct, count});
            total_correct += correct; total_count += count;
        }
        sort(reports.begin(), reports.end(), [](const Report&a, const Report&b){ return a.subject < b.subject; });
        float macro = 0.0f;
        for (auto& r : reports) macro += (r.total>0 ? float(r.correct)/float(r.total) : 0.0f);
        if (!reports.empty()) macro /= float(reports.size());
        float micro = (total_count>0) ? float(total_correct)/float(total_count) : 0.0f;

        cout << "Per-subject:" << endl;
        for (auto& r : reports) {
            printf("  %-30s | n=%4d | acc=%.2f%%\n", r.subject.c_str(), r.total, (r.total>0? 100.0f*float(r.correct)/float(r.total) : 0.0f));
        }
        printf("\nMacro=%.2f%% | Micro=%.2f%%\n", 100.0f*macro, 100.0f*micro);
        if (!args.out_file.empty()) {
            ofstream out(args.out_file, ios::app);
            if (out) {
                for (auto& r : reports) {
                    out << "{\"task\":\"mmlu\",\"subject\":\"" << r.subject
                        << "\",\"n\":" << r.total << ",\"acc\":" << (r.total>0? float(r.correct)/float(r.total):0.0f) << "}\n";
                }
                out << "{\"task\":\"mmlu\",\"macro\":" << macro << ",\"micro\":" << micro
                    << ",\"split\":\"" << args.split << "\",\"fewshot\":" << args.fewshot << "}\n";
                out.close();
            }
        }
        cout << "\n✅ Done." << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}


