#!/bin/zsh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PY="${PYTHON:-python3}"

PRETRAINED="$(cd "$(dirname "$0")" && pwd)/pretrained"
LORA_DIR="$(cd "$(dirname "$0")" && pwd)/outputs/lora_adapter"
MMLU_ROOT="$REPO_ROOT/data/mmlu/data"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/outputs"
mkdir -p "$OUT_DIR"

SPLIT="${SPLIT:-dev}"     # dev or test
FEWSHOT="${FEWSHOT:-0}"   # commonly 5 for test

# Use local C++ evaluation binary built in this directory
LOCAL_BIN="$(cd "$(dirname "$0")" && pwd)/build/eval_mmlu"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] 構建本地二進位..."
  "$(cd "$(dirname "$0")" && pwd)/build.sh"
fi

echo "[Eval] pretrained=$PRETRAINED"
echo "[Eval] lora_path=$LORA_DIR/lora_final.safetensors"
echo "[Eval] split=$SPLIT fewshot=$FEWSHOT"

"$LOCAL_BIN" \
  --mmlu_root "$MMLU_ROOT" \
  --split "$SPLIT" \
  --fewshot "$FEWSHOT" \
  --pretrained_dir "$PRETRAINED" \
  --lora_path "$LORA_DIR/lora_final.safetensors" \
  --out "$OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"

echo "[Eval] Results saved to: $OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"


