/**
 * @file eval_ppl.cpp
 * @brief WikiText-2 Perplexity Evaluation
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/graph/lora_injector.h"
#include "finetune_ops/graph/lora_saver.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/lm_loss.h"

using namespace std;
using namespace ops;

struct Args {
    string data_root;
    string split = "valid"; // train|valid|test
    string pretrained_dir;
    string lora_path;
    bool lora_merge = true;
    int seq_len = 1024;
    int batch_size = 1;
    int log_every = 50;
    string out_file;
    bool debug = false;
};

static void usage(const char* prog) {
    cerr << "Usage: " << prog <<
        " --data_root PATH [--split valid|test] [--seq_len N] [--batch_size N]" << endl;
    cerr << "  [--pretrained_dir PATH] [--lora_path PATH] [--lora_merge 0|1]" << endl;
    cerr << "  [--out FILE] [--log_every N] [--debug 0|1]" << endl;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    a.data_root = "/Users/tony/Documents/restart/data/wikitext2/wikitext-2-raw";
    a.pretrained_dir = "/Users/tony/Documents/restart/gpt2_lora_finetune/pretrained/gpt2";
    
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get = [&](const string& key){ if (i+1>=argc || k!=key) { usage(argv[0]); exit(1);} return string(argv[++i]); };
        if (k == "--data_root") a.data_root = get("--data_root");
        else if (k == "--split") a.split = get("--split");
        else if (k == "--seq_len") a.seq_len = stoi(get("--seq_len"));
        else if (k == "--batch_size") a.batch_size = stoi(get("--batch_size"));
        else if (k == "--pretrained_dir") a.pretrained_dir = get("--pretrained_dir");
        else if (k == "--lora_path") a.lora_path = get("--lora_path");
        else if (k == "--lora_merge") a.lora_merge = (stoi(get("--lora_merge")) != 0);
        else if (k == "--out") a.out_file = get("--out");
        else if (k == "--log_every") a.log_every = stoi(get("--log_every"));
        else if (k == "--debug") a.debug = (stoi(get("--debug")) != 0);
        else if (k == "--help" || k == "-h") { usage(argv[0]); exit(0);} 
        else { cerr << "Unknown arg: " << k << endl; usage(argv[0]); exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    try {
        cout << "========== WikiText-2 PPL Evaluation ==========" << endl;
        cout << "data_root     : " << args.data_root << endl;
        cout << "split         : " << args.split << endl;
        cout << "seq_len       : " << args.seq_len << endl;
        cout << "pretrained_dir: " << args.pretrained_dir << endl;

        // 1) Load model
        GPT2Config cfg;
        try {
            cfg = GPT2Config::from_pretrained(args.pretrained_dir);
            cout << "  [OK] GPT-2 config loaded: layers=" << cfg.n_layer
                 << ", hidden=" << cfg.n_embd
                 << ", heads=" << cfg.n_head << endl;
        } catch (const std::exception& e) {
            cout << "  [WARN] Failed to read config, fallback to default: " << e.what() << endl;
        }
        if (args.seq_len > cfg.n_positions) {
            cout << "  [WARN] seq_len clipped from " << args.seq_len << " to " << cfg.n_positions << endl;
            args.seq_len = cfg.n_positions;
        }
        GPT2Model model(cfg);
        model.tie_weights();
        SafeTensorsReader reader(args.pretrained_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        // GPT-2 safetensors weights are in [in, out] format, consistent with model internal, no transpose needed
        SafeTensorsLoadOptions load_opts;
        load_opts.transpose_linear = false;
        load_opts.verbose = false;
        auto loaded = reader.load_tensors_mapped(mapping, load_opts);
        for (const auto& kv : loaded) {
            model.assign_weight(kv.first, kv.second);
        }

        // 2) LoRA (optional)
        LoraInjector injector;
        if (!args.lora_path.empty()) {
            cout << "[LoRA] Loading from " << args.lora_path << endl;
            auto lora_state = LoraSaver::load_safetensors(args.lora_path);
            
            if (!lora_state.compatible_with(model)) {
                cerr << "[ERROR] LoRA incompatible with model" << endl;
                return 1;
            }
            
            // Initialize LoRALinear modules (don't inject, directly attach loaded weights)
            model.init_lora_modules();
            
            // Attach loaded weights
            LoraSaver::attach_from_state(model, lora_state);
            
            if (args.lora_merge) {
                injector.merge_all(model);
                cout << "[LoRA] Merged to base weights" << endl;
            }
        }

        // 3) Tokenizer
        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tok(tok_cfg);
        tok.load();

        // 4) Dataset
        WT2Config data_cfg;
        data_cfg.train_path = args.data_root + "/wiki.train.raw";
        data_cfg.valid_path = args.data_root + "/wiki.valid.raw";
        data_cfg.test_path  = args.data_root + "/wiki.test.raw";
        data_cfg.seq_len = args.seq_len;
        data_cfg.stride  = -1;
        data_cfg.eos_id  = 50256;
        data_cfg.drop_last = false;
        WikiText2Dataset dataset(data_cfg, &tok);
        
        Split split_enum;
        if (args.split == "train") split_enum = Split::Train;
        else if (args.split == "valid") split_enum = Split::Valid;
        else split_enum = Split::Test;
        dataset.load(split_enum);
        dataset.reset_cursor();

        // 5) Evaluation
        cout << "[Eval] Computing perplexity..." << endl;
        double sum_loss = 0.0;
        int64_t sum_tokens = 0;
        size_t n_batches = 0;

        while (true) {
            auto batch = dataset.next_batch(args.batch_size, false);
            if (!batch.input_ids) break;
            
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            
            // Count valid tokens
            const int32_t* labels = batch.labels->data<int32_t>();
            int64_t valid = 0;
            for (int64_t i = 0; i < batch.labels->numel(); ++i) {
                if (labels[i] != -100) valid++;
            }
            
            sum_loss += loss_val * valid;
            sum_tokens += valid;
            n_batches++;
            
            // Progress output
            if (args.log_every > 0 && n_batches % args.log_every == 0) {
                float cur_nll = (sum_tokens > 0) ? (sum_loss / sum_tokens) : 0.0f;
                float cur_ppl = exp(cur_nll);
                printf("[progress] batches=%ld tokens=%lld nll=%.4f ppl=%.2f\n", 
                       n_batches, (long long)sum_tokens, cur_nll, cur_ppl);
                
                // Optional: Append to output file
                if (!args.out_file.empty()) {
                    ofstream out(args.out_file, ios::app);
                    if (out) {
                        out << "{\"task\":\"wt2_ppl_progress\",\"batches\":" << n_batches 
                            << ",\"tokens\":" << sum_tokens 
                            << ",\"nll\":" << cur_nll << ",\"ppl\":" << cur_ppl << "}\n";
                        out.close();
                    }
                }
            }
        }

        float mean_nll = (sum_tokens > 0) ? (sum_loss / sum_tokens) : 0.0f;
        float ppl = exp(mean_nll);

        cout << "\n\n========== Results ==========" << endl;
        printf("split=%s  tokens=%lld  nll=%.4f  ppl=%.2f\n", 
               args.split.c_str(), (long long)sum_tokens, mean_nll, ppl);

        // Write to file
        if (!args.out_file.empty()) {
            ofstream out(args.out_file, ios::app);
            if (out) {
                out << "{\"task\":\"wt2_ppl\",\"split\":\"" << args.split 
                    << "\",\"tokens\":" << sum_tokens 
                    << ",\"nll\":" << mean_nll 
                    << ",\"ppl\":" << ppl 
                    << ",\"model\":\"gpt2\",\"lora\":\"" << args.lora_path << "\"}\n";
                out.close();
                cout << "[OK] Wrote results to " << args.out_file << endl;
            }
        }

        // Unmerge
        if (!args.lora_path.empty() && args.lora_merge) {
            injector.unmerge_all(model);
        }

        cout << "\n[DONE] Completed." << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n[ERROR] Exception: " << e.what() << endl;
        return 1;
    }
}
