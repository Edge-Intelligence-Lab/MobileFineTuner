#!/usr/bin/env python3
import argparse
import json
import math
import os
import struct
from typing import List

import torch
from safetensors.torch import load_file
import sentencepiece as spm


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * norm) * (1.0 + weight)


def rotate_half(x: torch.Tensor):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos, sin):
    # cos/sin: [seq, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def build_rope(seq_len: int, head_dim: int, theta: float, device):
    half = head_dim // 2
    inv_freq = torch.pow(
        theta,
        -torch.arange(0, half, dtype=torch.float32, device=device) / half,
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)[:, None]
    freqs = positions * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def build_mask(seq_len: int, sliding_window: int = None, device="cpu"):
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    i = torch.arange(seq_len, device=device).view(1, 1, -1, 1)
    j = torch.arange(seq_len, device=device).view(1, 1, 1, -1)
    causal = (j <= i)
    if sliding_window is not None:
        window = (i - j) < sliding_window
        allow = causal & window
    else:
        allow = causal
    mask = mask.masked_fill(~allow, -1e10)
    return mask


class GemmaReferenceModel:
    def __init__(self, model_dir: str, device="cpu"):
        self.device = device
        cfg_path = os.path.join(model_dir, "config.json")
        self.cfg = load_config(cfg_path)
        self.hidden_size = self.cfg["hidden_size"]
        self.intermediate = self.cfg["intermediate_size"]
        self.num_layers = self.cfg["num_hidden_layers"]
        self.num_heads = self.cfg["num_attention_heads"]
        self.num_kv = self.cfg["num_key_value_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.layer_types = self.cfg["layer_types"]
        self.sliding_window = self.cfg["sliding_window"]
        self.rms_eps = self.cfg["rms_norm_eps"]
        self.scaling = self.cfg["query_pre_attn_scalar"] ** -0.5

        weights_path = os.path.join(model_dir, "model.safetensors")
        raw = load_file(weights_path)
        self.weights = {k: v.to(torch.float32).to(device) for k, v in raw.items()}

    def embed(self, input_ids: torch.Tensor):
        weight = self.weights["model.embed_tokens.weight"]
        embeds = torch.nn.functional.embedding(input_ids, weight)
        return embeds * math.sqrt(self.hidden_size)

    def mlp(self, x, i):
        gate_w = self.weights[f"model.layers.{i}.mlp.gate_proj.weight"]
        up_w = self.weights[f"model.layers.{i}.mlp.up_proj.weight"]
        down_w = self.weights[f"model.layers.{i}.mlp.down_proj.weight"]
        gate = torch.matmul(x, gate_w.T)
        up = torch.matmul(x, up_w.T)
        act_gate = torch.nn.functional.gelu(gate, approximate="tanh")
        out = act_gate * up
        out = torch.matmul(out, down_w.T)
        return out

    def attention(self, x, i, cos, sin, mask):
        q_w = self.weights[f"model.layers.{i}.self_attn.q_proj.weight"]
        k_w = self.weights[f"model.layers.{i}.self_attn.k_proj.weight"]
        v_w = self.weights[f"model.layers.{i}.self_attn.v_proj.weight"]
        o_w = self.weights[f"model.layers.{i}.self_attn.o_proj.weight"]

        q = torch.matmul(x, q_w.T)
        k = torch.matmul(x, k_w.T)
        v = torch.matmul(x, v_w.T)

        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv, self.head_dim).transpose(1, 2)

        q_norm = self.weights[f"model.layers.{i}.self_attn.q_norm.weight"]
        k_norm = self.weights[f"model.layers.{i}.self_attn.k_norm.weight"]
        q = rms_norm(q, q_norm, self.rms_eps)
        k = rms_norm(k, k_norm, self.rms_eps)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kv = self.num_heads // self.num_kv
        k = k.repeat_interleave(kv, dim=1)
        v = v.repeat_interleave(kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        out = torch.matmul(context, o_w.T)
        return out

    def block(self, hidden, i, cos, sin, mask):
        inp_ln = self.weights[f"model.layers.{i}.input_layernorm.weight"]
        post_attn_ln = self.weights[f"model.layers.{i}.post_attention_layernorm.weight"]
        pre_ff_ln = self.weights[f"model.layers.{i}.pre_feedforward_layernorm.weight"]
        post_ff_ln = self.weights[f"model.layers.{i}.post_feedforward_layernorm.weight"]

        residual = hidden
        hidden = rms_norm(hidden, inp_ln, self.rms_eps)
        attn_out = self.attention(hidden, i, cos, sin, mask)
        attn_out = rms_norm(attn_out, post_attn_ln, self.rms_eps)
        hidden = residual + attn_out

        residual = hidden
        hidden = rms_norm(hidden, pre_ff_ln, self.rms_eps)
        mlp_out = self.mlp(hidden, i)
        mlp_out = rms_norm(mlp_out, post_ff_ln, self.rms_eps)
        hidden = residual + mlp_out
        return hidden

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed(input_ids)
        B, S = hidden.shape[0], hidden.shape[1]
        cosine_global, sine_global = build_rope(
            S, self.head_dim, self.cfg["rope_theta"], device=self.device
        )
        cosine_local, sine_local = build_rope(
            S, self.head_dim, self.cfg["rope_local_base_freq"], device=self.device
        )
        causal_mask = build_mask(S, None, self.device)
        sliding_mask = build_mask(S, self.sliding_window, self.device)

        for idx in range(self.num_layers):
            is_sliding = self.layer_types[idx] == "sliding_attention"
            cos = cosine_local if is_sliding else cosine_global
            sin = sine_local if is_sliding else sine_global
            mask = sliding_mask if is_sliding else causal_mask
            hidden = self.block(hidden, idx, cos, sin, mask)

        final_norm = self.weights["model.norm.weight"]
        hidden = rms_norm(hidden, final_norm, self.rms_eps)
        lm_head = self.weights.get("lm_head.weight")
        if lm_head is None:
            lm_head = self.weights["model.embed_tokens.weight"]
        logits = torch.matmul(hidden, lm_head.T)
        return logits


def save_binary(samples, path: str, vocab_size: int):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(samples)))
        for sample in samples:
            text_bytes = sample["text"].encode("utf-8")
            f.write(struct.pack("<I", len(text_bytes)))
            f.write(text_bytes)
            seq = sample["input_ids"]
            f.write(struct.pack("<I", len(seq)))
            f.write(struct.pack(f"<{len(seq)}i", *seq))
            logits = sample["logits"]
            f.write(struct.pack("<I", len(logits)))
            f.write(struct.pack(f"<{len(logits)}f", *logits))


def main():
    parser = argparse.ArgumentParser(description="Generate Gemma golden logits")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Hello, Gemma!\n",
            "今天天气不错🙂",
        ],
    )
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(os.path.join(args.model_dir, "tokenizer.model"))

    device = "cpu"
    model = GemmaReferenceModel(args.model_dir, device=device)

    samples = []
    for text in args.prompts:
        ids = tokenizer.EncodeAsIds(text)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model.forward(input_ids)
        last_logits = logits[0, -1].cpu().tolist()
        samples.append({
            "text": text,
            "input_ids": ids,
            "logits": last_logits,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_binary(samples, args.output, model.cfg["vocab_size"])
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
