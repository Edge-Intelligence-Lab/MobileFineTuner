#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare masked JSONL for multiple-choice tasks.")
    p.add_argument("--task", required=True, choices=["arcc", "arce", "hellaswag", "piqa", "mmlu"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    return p.parse_args()


def task_spec(task: str):
    if task == "arcc":
        return {
            "dataset": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "splits": {"train": "train", "validation": "valid", "test": "test"},
        }
    if task == "arce":
        return {
            "dataset": "allenai/ai2_arc",
            "config": "ARC-Easy",
            "splits": {"train": "train", "validation": "valid", "test": "test"},
        }
    if task == "hellaswag":
        return {
            "dataset": "Rowan/hellaswag",
            "config": None,
            "splits": {"train": "train", "validation": "valid"},
        }
    if task == "piqa":
        return {
            "dataset": "piqa",
            "config": None,
            "splits": {"train": "train", "validation": "valid"},
        }
    if task == "mmlu":
        return {
            "dataset": "cais/mmlu",
            "config": "all",
            "splits": {"auxiliary_train": "train", "validation": "valid"},
        }
    raise KeyError(task)


def normalize_arc(ex):
    choices = ex.get("choices", {})
    labels = [str(x).strip() for x in choices.get("label", [])]
    texts = [str(x).strip() for x in choices.get("text", [])]
    if len(texts) != 4:
        return None
    answer_key = str(ex.get("answerKey", "")).strip()
    answer_idx = None
    if answer_key in labels:
        answer_idx = labels.index(answer_key)
    elif answer_key.upper() in LETTERS[:4]:
        answer_idx = LETTERS.index(answer_key.upper())
    elif answer_key.isdigit() and 1 <= int(answer_key) <= 4:
        answer_idx = int(answer_key) - 1
    if answer_idx is None or answer_idx >= 4:
        return None
    prompt = (
        f"Question: {str(ex.get('question', '')).strip()}\n"
        f"A. {texts[0]}\n"
        f"B. {texts[1]}\n"
        f"C. {texts[2]}\n"
        f"D. {texts[3]}\n"
        "Answer: "
    )
    return prompt, LETTERS[answer_idx]


def normalize_hellaswag(ex):
    endings = [str(x).strip() for x in ex.get("endings", [])]
    if len(endings) != 4:
        return None
    label = str(ex.get("label", "")).strip()
    if not label.isdigit():
        return None
    idx = int(label)
    if idx < 0 or idx >= 4:
        return None
    context = str(ex.get("ctx", "")).strip()
    if not context:
        a = str(ex.get("ctx_a", "")).strip()
        b = str(ex.get("ctx_b", "")).strip()
        context = (a + " " + b).strip()
    prompt = (
        f"Context: {context}\n"
        f"A. {endings[0]}\n"
        f"B. {endings[1]}\n"
        f"C. {endings[2]}\n"
        f"D. {endings[3]}\n"
        "Answer: "
    )
    return prompt, LETTERS[idx]


def normalize_piqa(ex):
    sol1 = str(ex.get("sol1", "")).strip()
    sol2 = str(ex.get("sol2", "")).strip()
    label = ex.get("label")
    if label not in (0, 1, "0", "1"):
        return None
    idx = int(label)
    prompt = (
        f"Goal: {str(ex.get('goal', '')).strip()}\n"
        f"A. {sol1}\n"
        f"B. {sol2}\n"
        "Answer: "
    )
    return prompt, LETTERS[idx]


def normalize_mmlu(ex):
    choices = [str(x).strip() for x in ex.get("choices", [])]
    if len(choices) != 4:
        return None
    answer = ex.get("answer")
    if not isinstance(answer, int) or answer < 0 or answer >= 4:
        return None
    prompt = (
        f"Question: {str(ex.get('question', '')).strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer: "
    )
    return prompt, LETTERS[answer]


def normalize_example(task: str, ex):
    if task in {"arcc", "arce"}:
        return normalize_arc(ex)
    if task == "hellaswag":
        return normalize_hellaswag(ex)
    if task == "piqa":
        return normalize_piqa(ex)
    if task == "mmlu":
        return normalize_mmlu(ex)
    raise KeyError(task)


def build_record(tokenizer, prompt: str, answer_letter: str, seq_len: int, eos_id: int):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer_letter, add_special_tokens=False)
    ids = prompt_ids + answer_ids + [eos_id]
    if len(ids) > seq_len:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids) + [0]
    pad = seq_len - len(ids)
    if pad:
        ids.extend([eos_id] * pad)
        mask.extend([0] * pad)
    return {"ids": ids, "mask": mask}


def load_split(spec, hf_split):
    kwargs = {"split": hf_split}
    if spec["dataset"] == "piqa":
        kwargs["trust_remote_code"] = True
    if spec["config"] is None:
        return load_dataset(spec["dataset"], **kwargs)
    return load_dataset(spec["dataset"], spec["config"], **kwargs)


def write_split(task: str, tokenizer, dataset_split, out_path: Path, seq_len: int, eos_id: int):
    total = kept = skipped_shape = skipped_len = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in dataset_split:
            total += 1
            norm = normalize_example(task, ex)
            if norm is None:
                skipped_shape += 1
                continue
            prompt, answer_letter = norm
            rec = build_record(tokenizer, prompt, answer_letter, seq_len, eos_id)
            if rec is None:
                skipped_len += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return {
        "total": total,
        "kept": kept,
        "skipped_nonstandard": skipped_shape,
        "skipped_over_seq_len": skipped_len,
    }


def main():
    args = parse_args()
    spec = task_spec(args.task)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has neither eos_token_id nor pad_token_id")

    manifest = {
        "task": args.task,
        "dataset": spec["dataset"],
        "config": spec["config"],
        "model_dir": str(Path(args.model_dir).resolve()),
        "seq_len": args.seq_len,
        "eos_id": int(eos_id),
        "splits": {},
    }
    for hf_split, out_name in spec["splits"].items():
        ds = load_split(spec, hf_split)
        stats = write_split(args.task, tokenizer, ds, out_dir / f"{out_name}.jsonl", args.seq_len, int(eos_id))
        manifest["splits"][out_name] = stats
        print(out_name, stats)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
