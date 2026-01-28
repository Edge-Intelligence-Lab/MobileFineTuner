#!/usr/bin/env python3
"""
Generate reference logits from PyTorch/HuggingFace GPT-2 (for C++ forward alignment)
"""

import torch
import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def main():
    # Load pretrained model (dropout disabled)
    tok = GPT2TokenizerFast.from_pretrained("/path/to/gpt2_lora_finetune/pretrained/gpt2")
    model = GPT2LMHeadModel.from_pretrained("/path/to/gpt2_lora_finetune/pretrained/gpt2")
    model.eval()
    
    # Fixed input
    text = "Hello, world!\n"
    x = tok(text, return_tensors="pt")
    
    print(f"Input text: {repr(text)}")
    print(f"Input IDs: {x['input_ids'].tolist()}")
    print(f"Attention mask: {x['attention_mask'].tolist()}")
    
    # Forward pass (dropout disabled)
    with torch.no_grad():
        logits = model(**x).logits  # [1, S, V]
    
    # Save logits of the last token
    last_logits = logits[0, -1].float().cpu()  # [V]
    
    # Top-5
    topv, topi = torch.topk(last_logits, 5)
    print(f"\nPyTorch top-5 IDs:  {topi.tolist()}")
    print(f"PyTorch top-5 vals: {[f'{v:.6f}' for v in topv.tolist()]}")
    
    # Save as JSON (easy to read from C++)
    output_path = "/path/to/operators/finetune_ops/graph/pt_last_logits.json"
    with open(output_path, "w") as f:
        json.dump(last_logits.tolist(), f)
    
    print(f"\nSaved PyTorch logits to: {output_path}")
    print(f"Logits shape: {last_logits.shape}, dtype: {last_logits.dtype}")
    print(f"Argmax: {last_logits.argmax().item()}")

if __name__ == "__main__":
    main()

