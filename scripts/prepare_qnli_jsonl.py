#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from transformers import AutoTokenizer

try:
    from asset_paths import resolve_model_dir, repo_root_from
except ImportError:
    import sys

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))
    from asset_paths import resolve_model_dir, repo_root_from


QNLI_URL = "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip"
QNLI_MD5 = "b4efd6554440de1712e9b54e14760e82"
TASK_INSTRUCTION = (
    "Task: Determine whether the sentence contains the answer to the question."
)


@dataclass
class SplitStats:
    total: int = 0
    kept: int = 0
    truncated: int = 0
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "truncated": self.truncated,
            "dropped": self.dropped,
        }


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_qnli(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "QNLIv2.zip"
    if zip_path.is_file() and md5sum(zip_path) == QNLI_MD5:
        return zip_path
    urllib.request.urlretrieve(QNLI_URL, zip_path)
    if md5sum(zip_path) != QNLI_MD5:
        raise RuntimeError(f"QNLI md5 mismatch for {zip_path}")
    return zip_path


def extract_qnli(zip_path: Path, cache_dir: Path) -> Path:
    extracted = cache_dir / "QNLI"
    train_tsv = extracted / "train.tsv"
    dev_tsv = extracted / "dev.tsv"
    if train_tsv.is_file() and dev_tsv.is_file():
        return extracted
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    if not train_tsv.is_file():
        raise RuntimeError(f"Expected extracted QNLI train split at {train_tsv}")
    return extracted


def truncate_pair(
    question_ids: List[int],
    sentence_ids: List[int],
    available_tokens: int,
    min_sentence_budget: int = 16,
) -> Tuple[List[int], List[int], bool]:
    total = len(question_ids) + len(sentence_ids)
    if total <= available_tokens:
        return question_ids, sentence_ids, False
    if available_tokens <= 0:
        return [], [], True

    keep_question = min(len(question_ids), max(8, available_tokens - min_sentence_budget))
    keep_question = min(keep_question, available_tokens)
    trimmed_question = question_ids[:keep_question]
    remaining = max(0, available_tokens - len(trimmed_question))
    trimmed_sentence = sentence_ids[:remaining]

    if not trimmed_sentence and available_tokens > 0:
        sentence_budget = min(len(sentence_ids), available_tokens // 2)
        trimmed_sentence = sentence_ids[:sentence_budget]
        remaining = max(0, available_tokens - len(trimmed_sentence))
        trimmed_question = question_ids[:remaining]

    return trimmed_question, trimmed_sentence, True


def build_record(
    tokenizer,
    question: str,
    sentence: str,
    label_text: str,
    seq_len: int,
    eos_id: int,
) -> Tuple[dict | None, bool]:
    instr_ids = tokenizer.encode(f"{TASK_INSTRUCTION}\nQuestion: ", add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    sentence_prefix_ids = tokenizer.encode("\nSentence: ", add_special_tokens=False)
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    answer_prefix_ids = tokenizer.encode("\nAnswer:", add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + label_text, add_special_tokens=False)

    reserved = len(instr_ids) + len(sentence_prefix_ids) + len(answer_prefix_ids) + len(answer_ids) + 1
    available = seq_len - reserved
    if available <= 0:
        return None, False

    question_ids, sentence_ids, truncated = truncate_pair(question_ids, sentence_ids, available)
    prompt_ids = instr_ids + question_ids + sentence_prefix_ids + sentence_ids + answer_prefix_ids
    ids = prompt_ids + answer_ids + [eos_id]
    if len(ids) > seq_len:
        return None, truncated

    mask = [0] * len(prompt_ids) + [1] * len(answer_ids) + [0]
    return {"ids": ids, "mask": mask}, truncated


def iter_tsv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def convert_split(
    tokenizer,
    input_path: Path,
    output_path: Path,
    seq_len: int,
    eos_id: int,
    max_samples: int,
) -> SplitStats:
    stats = SplitStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_tsv_rows(input_path):
            stats.total += 1
            label = row.get("label")
            if label not in {"entailment", "not_entailment"}:
                continue
            label_text = "yes" if label == "entailment" else "no"
            record, truncated = build_record(
                tokenizer,
                row["question"].strip(),
                row["sentence"].strip(),
                label_text,
                seq_len,
                eos_id,
            )
            if truncated:
                stats.truncated += 1
            if record is None:
                stats.dropped += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.kept += 1
            if max_samples > 0 and stats.kept >= max_samples:
                break
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QNLI masked-JSONL for Qwen causal-LM finetuning.")
    parser.add_argument("--model_dir", default="", help="Tokenizer/model directory. Defaults to resolved local Qwen dir.")
    parser.add_argument("--output_dir", default="", help="Output directory for train.jsonl/valid.jsonl/stats.json.")
    parser.add_argument("--cache_dir", default="", help="Where to cache/download QNLI.")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(repo_root_from(__file__))
    model_dir = args.model_dir or resolve_model_dir("qwen", repo_root / "qwen_lora_finetune/pretrained")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / f"data/qnli/qwen_qnli_s{args.seq_len}"
    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / "data/qnli/_cache"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer is missing eos_token_id")
    eos_id = int(tokenizer.eos_token_id)

    zip_path = download_qnli(cache_dir)
    qnli_dir = extract_qnli(zip_path, cache_dir)

    train_stats = convert_split(
        tokenizer,
        qnli_dir / "train.tsv",
        output_dir / "train.jsonl",
        args.seq_len,
        eos_id,
        args.max_train_samples,
    )
    valid_stats = convert_split(
        tokenizer,
        qnli_dir / "dev.tsv",
        output_dir / "valid.jsonl",
        args.seq_len,
        eos_id,
        args.max_valid_samples,
    )

    meta = {
        "task": "qnli",
        "format": "masked_jsonl",
        "label_texts": {"entailment": "yes", "not_entailment": "no"},
        "prompt_template": (
            "Task: Determine whether the sentence contains the answer to the question.\\n"
            "Question: {question}\\nSentence: {sentence}\\nAnswer: {yes|no}"
        ),
        "seq_len": args.seq_len,
        "model_dir": model_dir,
        "source_url": QNLI_URL,
        "train": train_stats.to_dict(),
        "valid": valid_stats.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": str(output_dir),
        "train_kept": train_stats.kept,
        "valid_kept": valid_stats.kept,
        "train_dropped": train_stats.dropped,
        "valid_dropped": valid_stats.dropped,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
