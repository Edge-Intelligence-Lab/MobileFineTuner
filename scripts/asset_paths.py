from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Sequence


def repo_root_from(anchor_file: str) -> Path:
    anchor = Path(anchor_file).resolve()
    for candidate in [anchor.parent, *anchor.parents]:
        if (candidate / "operator/CMakeLists.txt").is_file() and (candidate / "scripts").is_dir():
            return candidate
    return anchor.parent.parent


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def _first_dir_with_files(candidates: Sequence[Path], required_files: Sequence[str], description: str) -> str:
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / rel).is_file() for rel in required_files):
            return str(candidate)
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve {description}. Checked: {checked}")


def _has_model_weights(candidate: Path) -> bool:
    if (candidate / "model.safetensors").is_file():
        return True

    index_path = candidate / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    weight_map = index.get("weight_map", {})
    if not weight_map:
        return False
    return all((candidate / shard).is_file() for shard in set(weight_map.values()))


def _first_model_dir(candidates: Sequence[Path], required_files: Sequence[str], description: str) -> str:
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / rel).is_file() for rel in required_files) and _has_model_weights(candidate):
            return str(candidate)
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve {description}. Required: {', '.join(required_files)} plus "
        f"model.safetensors or a valid model.safetensors.index.json with shard files. Checked: {checked}"
    )


def resolve_wikitext_dir(repo_root: str | Path) -> str:
    root = Path(repo_root)
    data_root = _env_path("MFT_DATA_ROOT", "MOBILEFINETUNER_DATA_ROOT")
    candidates = []
    if data_root:
        candidates.extend([
            data_root / "wikitext2/wikitext-2-raw",
            data_root / "wikitext-2-raw",
        ])
    candidates.append(root / "data/wikitext2/wikitext-2-raw")
    return _first_dir_with_files(
        candidates,
        ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"],
        "WikiText-2 raw data",
    )


def resolve_mmlu_dir(repo_root: str | Path) -> str:
    root = Path(repo_root)
    data_root = _env_path("MFT_DATA_ROOT", "MOBILEFINETUNER_DATA_ROOT")
    candidates = []
    if data_root:
        candidates.extend([
            data_root / "mmlu/data",
            data_root / "mmlu",
        ])
    candidates.append(root / "data/mmlu/data")
    return _first_dir_with_files(
        candidates,
        ["README.txt", "dev/abstract_algebra_dev.csv"],
        "MMLU data",
    )


def resolve_model_dir(model_key: str, local_dir: str | Path) -> str:
    local = Path(local_dir)
    model_root = _env_path("MFT_MODEL_ROOT", "MOBILEFINETUNER_MODEL_ROOT")

    def root_candidates(*names: str) -> list[Path]:
        if not model_root:
            return []
        return [model_root / name for name in names]

    model_map = {
        "gpt2_small": (
            ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
            root_candidates("gpt2", "GPT2-124M", "gpt2-small") + [local],
        ),
        "gpt2_medium": (
            ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
            root_candidates("gpt2-medium", "GPT2-355M", "gpt2_medium") + [local],
        ),
        "gemma_270m": (
            ["config.json", "tokenizer.json", "tokenizer.model"],
            root_candidates("gemma-3-270m", "Gemma3-270M/gemma-3-270m", "Gemma3-270M") + [local],
        ),
        "gemma_1b_pt": (
            ["config.json", "tokenizer.json", "tokenizer.model"],
            root_candidates("gemma-3-1b-pt", "Gemma3-1B-PT/gemma-3-1b-pt", "Gemma3-1B-PT") + [local],
        ),
        "qwen": (
            ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
            root_candidates("Qwen2.5-0.5B", "Qwen3-0.6B", "qwen2.5-0.5b", "qwen") + [local],
        ),
    }
    if model_key not in model_map:
        raise KeyError(f"Unknown model key: {model_key}")
    required, candidates = model_map[model_key]
    return _first_model_dir(candidates, required, f"{model_key} model assets")
