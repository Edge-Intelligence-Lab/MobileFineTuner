#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"
PY="${PYTHON:-python3}"

PRETRAINED="${GPT2_MEDIUM_MODEL_DIR:-$(mf_resolve_model_dir gpt2_medium "$SCRIPT_DIR/pretrained")}"
LORA_PATH="${LORA_PATH:-$SCRIPT_DIR/outputs/lora_final.safetensors}"
MMLU_ROOT="${MMLU_ROOT:-$(mf_resolve_mmlu_dir "$REPO_ROOT")}"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

SPLIT="${SPLIT:-dev}"     # dev or test
FEWSHOT="${FEWSHOT:-0}"   # commonly 5 for test
DEVICE="${DEVICE:-cpu}"   # cpu/mps/cuda (not used by local C++ eval)

# Use local C++ evaluation binary built in this directory
LOCAL_BIN="$SCRIPT_DIR/build/eval_mmlu"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] 構建本地二進位..."
  "$SCRIPT_DIR/build.sh"
fi

echo "[Eval] pretrained=$PRETRAINED"
echo "[Eval] lora_path=$LORA_PATH"
echo "[Eval] split=$SPLIT fewshot=$FEWSHOT"

"$LOCAL_BIN" \
  --mmlu_root "$MMLU_ROOT" \
  --split "$SPLIT" \
  --fewshot "$FEWSHOT" \
  --pretrained_dir "$PRETRAINED" \
  --lora_path "$LORA_PATH" \
  --out "$OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"

echo "[Eval] Results saved to: $OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"
