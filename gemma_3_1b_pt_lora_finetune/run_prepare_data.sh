#!/bin/zsh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

echo "[Prepare] Generating Gemma 3 1B PT MMLU JSONL (seq_len=128)..."
"$PY" "$DIR/prepare_data.py"
echo "[Prepare] Done: $(cd "$DIR/.." && pwd)/runs/mmlu_jsonl_gemma1b_s128"

