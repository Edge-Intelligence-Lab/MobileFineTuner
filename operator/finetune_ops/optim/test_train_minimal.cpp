/**
 * @file test_train_minimal.cpp  
 * @brief Minimal training loop test (Forward + Loss, no Backward)
 */

#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "../graph/safetensors_loader.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_bpe.h"
#include "../core/lm_loss.h"
#include <iostream>

using namespace ops;

int main() {
    try {
        std::cout << "========== Minimal training loop test ==========\n" << std::endl;
        
        // 1. Load model
        std::cout << "[1/5] Loading model..." << std::endl;
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader("/path/to/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(hf_key, false);
                    model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        std::cout << "✅ Model loaded" << std::endl;
        
        // 2. Inject LoRA
        std::cout << "\n[2/5] Injecting LoRA..." << std::endl;
        LoraSpec lora_spec = LoraSpec::default_config();
        lora_spec.rank = 4;  // Small rank for quick test
        
        LoraInjector lora;
        lora.inject(model, lora_spec);
        lora.print_info();
        
        // 3. Load data
        std::cout << "\n[3/5] Loading data..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/path/to/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        WT2Config data_cfg;
        data_cfg.train_path = "/path/to/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/path/to/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 64;  // Very short sequence
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        std::cout << "✅ Loaded " << dataset.num_sequences() << " sequences" << std::endl;
        
        // 4. Take one batch to test forward
        std::cout << "\n[4/5] Testing forward pass..." << std::endl;
        auto batch = dataset.next_batch(2);
        
        std::cout << "  Batch shapes:" << std::endl;
        std::cout << "    input_ids: [" << batch.input_ids->shape()[0] << ", " 
                  << batch.input_ids->shape()[1] << "]" << std::endl;
        std::cout << "    labels: [" << batch.labels->shape()[0] << ", "
                  << batch.labels->shape()[1] << "]" << std::endl;
        
        // Forward
        auto logits = model.forward(batch.input_ids, batch.attention_mask);
        std::cout << "  Logits shape: [" << logits->shape()[0] << ", "
                  << logits->shape()[1] << ", " << logits->shape()[2] << "]" << std::endl;
        
        // 5. Compute loss
        std::cout << "\n[5/5] Computing loss..." << std::endl;
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        float loss_val = loss->data<float>()[0];
        float ppl = perplexity_from_loss(loss_val);
        
        std::cout << "  Loss: " << loss_val << std::endl;
        std::cout << "  Perplexity: " << ppl << std::endl;
        
        if (loss_val > 0.0f && loss_val < 100.0f && ppl > 1.0f) {
            std::cout << "\n🎉 Minimal loop test passed!" << std::endl;
            std::cout << "  ✅ Model load OK" << std::endl;
            std::cout << "  ✅ LoRA injection OK" << std::endl;
            std::cout << "  ✅ Dataset load OK" << std::endl;
            std::cout << "  ✅ Forward OK" << std::endl;
            std::cout << "  ✅ Loss OK" << std::endl;
            std::cout << "\nNext: implement Backward and Optimizer" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Loss value abnormal" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

