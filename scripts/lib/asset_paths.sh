#!/usr/bin/env bash

mf_dir_has_files() {
  local dir="$1"
  shift
  [ -d "$dir" ] || return 1
  local rel
  for rel in "$@"; do
    [ -f "$dir/$rel" ] || return 1
  done
}

mf_dir_has_model_weights() {
  local dir="$1"
  if [ -f "$dir/model.safetensors" ]; then
    return 0
  fi
  if [ ! -f "$dir/model.safetensors.index.json" ]; then
    return 1
  fi
  python3 - "$dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
try:
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
except Exception:
    sys.exit(1)

weight_map = index.get("weight_map", {})
if not weight_map:
    sys.exit(1)

missing = [name for name in sorted(set(weight_map.values())) if not (model_dir / name).is_file()]
sys.exit(1 if missing else 0)
PY
}

mf_print_candidates() {
  local first=1
  local candidate
  for candidate in "$@"; do
    if [ $first -eq 1 ]; then
      printf '%s' "$candidate"
      first=0
    else
      printf ', %s' "$candidate"
    fi
  done
  printf '\n'
}

mf_resolve_existing_dir() {
  local description="$1"
  shift
  local required=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    required+=("$1")
    shift
  done
  [ "$#" -gt 0 ] && shift

  local candidates=("$@")
  local candidate
  for candidate in "${candidates[@]}"; do
    if mf_dir_has_files "$candidate" "${required[@]}"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Could not resolve $description." >&2
  echo "Checked: $(mf_print_candidates "${candidates[@]}")" >&2
  return 1
}

mf_wikitext_candidates() {
  local repo_root="$1"
  local data_root="${MFT_DATA_ROOT:-${MOBILEFINETUNER_DATA_ROOT:-}}"
  if [ -n "$data_root" ]; then
    cat <<EOF
$data_root/wikitext2/wikitext-2-raw
$data_root/wikitext-2-raw
EOF
  fi
  cat <<EOF
$repo_root/data/wikitext2/wikitext-2-raw
EOF
}

mf_mmlu_candidates() {
  local repo_root="$1"
  local data_root="${MFT_DATA_ROOT:-${MOBILEFINETUNER_DATA_ROOT:-}}"
  if [ -n "$data_root" ]; then
    cat <<EOF
$data_root/mmlu/data
$data_root/mmlu
EOF
  fi
  cat <<EOF
$repo_root/data/mmlu/data
EOF
}

mf_model_candidates() {
  local model_key="$1"
  local local_dir="$2"
  local model_root="${MFT_MODEL_ROOT:-${MOBILEFINETUNER_MODEL_ROOT:-}}"
  mf_emit_model_root_candidates() {
    [ -n "$model_root" ] || return 0
    for name in "$@"; do
      printf '%s\n' "$model_root/$name"
    done
  }
  case "$model_key" in
    gpt2_small)
      mf_emit_model_root_candidates gpt2 GPT2-124M gpt2-small
      cat <<EOF
$local_dir
EOF
      ;;
    gpt2_medium)
      mf_emit_model_root_candidates gpt2-medium GPT2-355M gpt2_medium
      cat <<EOF
$local_dir
EOF
      ;;
    gemma_270m)
      mf_emit_model_root_candidates gemma-3-270m Gemma3-270M/gemma-3-270m Gemma3-270M
      cat <<EOF
$local_dir
EOF
      ;;
    gemma_1b_pt)
      mf_emit_model_root_candidates gemma-3-1b-pt Gemma3-1B-PT/gemma-3-1b-pt Gemma3-1B-PT
      cat <<EOF
$local_dir
EOF
      ;;
    qwen)
      mf_emit_model_root_candidates Qwen2.5-0.5B Qwen3-0.6B qwen2.5-0.5b qwen
      cat <<EOF
$local_dir
EOF
      ;;
    *)
      echo "Unknown model key: $model_key" >&2
      return 1
      ;;
  esac
}

mf_resolve_wikitext_dir() {
  local repo_root="$1"
  local candidates=()
  local line
  while IFS= read -r line; do
    candidates+=("$line")
  done < <(mf_wikitext_candidates "$repo_root")
  mf_resolve_existing_dir "WikiText-2 raw data" \
    wiki.train.raw wiki.valid.raw wiki.test.raw -- "${candidates[@]}"
}

mf_resolve_mmlu_dir() {
  local repo_root="$1"
  local candidates=()
  local line
  while IFS= read -r line; do
    candidates+=("$line")
  done < <(mf_mmlu_candidates "$repo_root")
  mf_resolve_existing_dir "MMLU data" \
    README.txt dev/abstract_algebra_dev.csv -- "${candidates[@]}"
}

mf_resolve_model_dir() {
  local model_key="$1"
  local local_dir="$2"
  local required=()
  case "$model_key" in
    gpt2_small|gpt2_medium|qwen)
      required=(config.json tokenizer.json vocab.json merges.txt)
      ;;
    gemma_270m|gemma_1b_pt)
      required=(config.json tokenizer.json tokenizer.model)
      ;;
    *)
      echo "Unknown model key: $model_key" >&2
      return 1
      ;;
  esac

  local candidates=()
  local line
  while IFS= read -r line; do
    candidates+=("$line")
  done < <(mf_model_candidates "$model_key" "$local_dir")

  local candidate
  for candidate in "${candidates[@]}"; do
    if mf_dir_has_files "$candidate" "${required[@]}" && mf_dir_has_model_weights "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Could not resolve $model_key model assets." >&2
  echo "Required: ${required[*]} plus model.safetensors or a valid model.safetensors.index.json with shard files" >&2
  echo "Checked: $(mf_print_candidates "${candidates[@]}")" >&2
  return 1
}
