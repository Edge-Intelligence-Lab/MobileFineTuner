#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/graph/lora_injector.h"
#include "finetune_ops/graph/lora_saver.h"
#include "finetune_ops/core/tokenizer_bpe.h"
#include "mmlu/mmlu_runner.h"

using namespace std;
using namespace ops;

struct Args {
    string mmlu_root;
    string split = "dev"; // dev|test
    string pretrained_dir;
    string lora_path;
    bool lora_merge = true;
    int fewshot = 0;
    string out_file;
    bool debug = false;
};

static void usage(const char* prog) {
    cerr << "Usage: " << prog <<
        " --mmlu_root PATH [--split dev|test] [--fewshot K] [--pretrained_dir PATH]" << endl;
    cerr << "  [--lora_path PATH] [--lora_merge 0|1] [--out FILE] [--debug 0|1]" << endl;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    a.mmlu_root = "data/mmlu/data";
    a.pretrained_dir = "pretrained";
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get = [&](const string& key){ if (i+1>=argc || k!=key) { usage(argv[0]); exit(1);} return string(argv[++i]); };
        if (k == "--mmlu_root") a.mmlu_root = get("--mmlu_root");
        else if (k == "--split") a.split = get("--split");
        else if (k == "--fewshot") a.fewshot = stoi(get("--fewshot"));
        else if (k == "--pretrained_dir") a.pretrained_dir = get("--pretrained_dir");
        else if (k == "--lora_path") a.lora_path = get("--lora_path");
        else if (k == "--lora_merge") a.lora_merge = (stoi(get("--lora_merge")) != 0);
        else if (k == "--out") a.out_file = get("--out");
        else if (k == "--debug") a.debug = (stoi(get("--debug")) != 0);
        else if (k == "--help" || k == "-h") { usage(argv[0]); exit(0);} 
        else { cerr << "Unknown arg: " << k << endl; usage(argv[0]); exit(1); }
    }
    if (a.split != "dev" && a.split != "test") {
        cerr << "Invalid --split: " << a.split << ", must be dev|test" << endl;
        exit(1);
    }
    return a;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    try {
        cout << "========== MMLU Evaluation (GPT-2) ==========" << endl;
        cout << "mmlu_root     : " << args.mmlu_root << endl;
        cout << "split         : " << args.split << endl;
        cout << "fewshot       : " << args.fewshot << endl;
        cout << "pretrained_dir: " << args.pretrained_dir << endl;

        GPT2Config cfg = GPT2Config::from_pretrained(args.pretrained_dir + "/config.json");
        GPT2Model model(cfg);
        model.tie_weights();
        SafeTensorsReader reader(args.pretrained_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        for (const auto& kv : mapping) {
            const string& ik = kv.first; const string& hk = kv.second;
            try {
                auto info = reader.get_tensor_info(hk);
                if (!info.dtype.empty()) {
                    auto t = reader.load_tensor(hk, false);
                    model.assign_weight(ik, t);
                }
            } catch (...) {}
        }

        LoraInjector injector;
        if (!args.lora_path.empty()) {
            cout << "[LoRA] Loading from " << args.lora_path << endl;
            auto lora_state = LoraSaver::load_safetensors(args.lora_path);
            if (!lora_state.compatible_with(model)) {
                cerr << "❌ LoRA incompatible with model" << endl;
                return 1;
            }
            model.init_lora_modules();
            LoraSaver::attach_from_state(model, lora_state);
            if (args.lora_merge) {
                injector.merge_all(model);
                cout << "[LoRA] Merged to base weights" << endl;
            }
        }

        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tok(tok_cfg);
        tok.load();

        cout << "[Eval] Loading MMLU CSV..." << endl;
        MMLURunner runner(args.mmlu_root, args.split);
        runner.load();
        cout << "[Eval] Loaded. Running evaluation..." << endl;
        auto result = runner.evaluate(model, tok, args.fewshot);
        cout << "[Eval] Evaluation finished." << endl;

        cout << "\nPer-subject:" << endl;
        for (const auto& r : result.per_subject) {
            printf("  %-30s | n=%4d | acc=%.2f%%\n", r.subject.c_str(), r.total, r.accuracy()*100.0f);
        }
        printf("\nMacro=%.2f%% | Micro=%.2f%%\n", result.macro*100.0f, result.micro*100.0f);
        
        if (!args.out_file.empty()) {
            ofstream out(args.out_file, ios::app);
            if (out) {
                for (const auto& r : result.per_subject) {
                    out << "{\"task\":\"mmlu\",\"subject\":\"" << r.subject 
                        << "\",\"n\":" << r.total << ",\"acc\":" << r.accuracy() << "}\n";
                }
                out << "{\"task\":\"mmlu\",\"macro\":" << result.macro 
                    << ",\"micro\":" << result.micro 
                    << ",\"split\":\"" << args.split << "\",\"fewshot\":" << args.fewshot 
                    << ",\"lora\":\"" << args.lora_path << "\"}\n";
                out.close();
                cout << "✅ Wrote results to " << args.out_file << endl;
            }
        }
        
        if (!args.lora_path.empty() && args.lora_merge) {
            injector.unmerge_all(model);
        }
        cout << "\n✅ Done." << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}


