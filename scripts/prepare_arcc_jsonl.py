#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare ARC-Challenge masked JSONL for MF.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--dataset_config", default="ARC-Challenge")
    return p.parse_args()


def normalize_example(ex):
    choices = ex.get("choices", {})
    labels = [str(x).strip() for x in choices.get("label", [])]
    texts = [str(x).strip() for x in choices.get("text", [])]
    if len(texts) != 4:
        return None

    answer_key = str(ex.get("answerKey", "")).strip()
    if not answer_key:
        return None

    answer_idx = None
    if answer_key in labels:
        answer_idx = labels.index(answer_key)
    elif answer_key.upper() in LETTERS[:4]:
        answer_idx = LETTERS.index(answer_key.upper())
    elif answer_key.isdigit() and 1 <= int(answer_key) <= 4:
        answer_idx = int(answer_key) - 1

    if answer_idx is None or answer_idx >= 4:
        return None

    return {
        "question": str(ex.get("question", "")).strip(),
        "choices": texts,
        "answer": LETTERS[answer_idx],
    }


def build_record(tokenizer, ex, seq_len, eos_id):
    prompt = (
        f"Question: {ex['question']}\n"
        f"A. {ex['choices'][0]}\n"
        f"B. {ex['choices'][1]}\n"
        f"C. {ex['choices'][2]}\n"
        f"D. {ex['choices'][3]}\n"
        "Answer: "
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + ex["answer"], add_special_tokens=False)
    ids = prompt_ids + answer_ids + [eos_id]
    if len(ids) > seq_len:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids) + [0]
    pad = seq_len - len(ids)
    if pad:
        ids.extend([eos_id] * pad)
        mask.extend([0] * pad)
    return {"ids": ids, "mask": mask}


def write_split(tokenizer, dataset_split, out_path, seq_len, eos_id):
    total = kept = skipped_shape = skipped_len = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in dataset_split:
            total += 1
            norm = normalize_example(ex)
            if norm is None:
                skipped_shape += 1
                continue
            rec = build_record(tokenizer, norm, seq_len, eos_id)
            if rec is None:
                skipped_len += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return {
        "total": total,
        "kept": kept,
        "skipped_non4_or_no_answer": skipped_shape,
        "skipped_over_seq_len": skipped_len,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has neither eos_token_id nor pad_token_id")

    mapping = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }
    manifest = {
        "dataset": "allenai/ai2_arc",
        "config": args.dataset_config,
        "model_dir": str(Path(args.model_dir).resolve()),
        "seq_len": args.seq_len,
        "eos_id": int(eos_id),
        "splits": {},
    }
    for hf_split, out_name in mapping.items():
        ds = load_dataset("allenai/ai2_arc", args.dataset_config, split=hf_split)
        stats = write_split(tokenizer, ds, out_dir / f"{out_name}.jsonl", args.seq_len, int(eos_id))
        manifest["splits"][out_name] = stats
        print(out_name, stats)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
