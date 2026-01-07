#!/bin/zsh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

echo "[Prepare] Generating GPT-2 Medium MMLU JSONL (seq_len=128)..."
"$PY" "$DIR/prepare_data.py"
echo "[Prepare] Done: $(cd "$DIR/.." && pwd)/runs/mmlu_jsonl_gpt2_medium_s128"

