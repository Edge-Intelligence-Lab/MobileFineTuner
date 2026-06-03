#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"
PY="${PYTHON:-python3}"

PRETRAINED="${GEMMA_1B_PT_MODEL_DIR:-$(mf_resolve_model_dir gemma_1b_pt "$SCRIPT_DIR/pretrained")}"
OUT_DIR="$SCRIPT_DIR/outputs"
MMLU_ROOT="${MMLU_ROOT:-$(mf_resolve_mmlu_dir "$REPO_ROOT")}"
mkdir -p "$OUT_DIR"

SPLIT="${SPLIT:-dev}"     # dev or test
FEWSHOT="${FEWSHOT:-0}"   # commonly 5 for test

# Use local C++ evaluation binary (Gemma)
LOCAL_BIN="$SCRIPT_DIR/build/eval_mmlu_gemma"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] 構建本地二進位..."
  "$SCRIPT_DIR/build.sh"
fi

echo "[Eval] pretrained=$PRETRAINED"
echo "[Eval] split=$SPLIT fewshot=$FEWSHOT"

"$LOCAL_BIN" \
  --mmlu_root "$MMLU_ROOT" \
  --split "$SPLIT" \
  --fewshot "$FEWSHOT" \
  --pretrained_dir "$PRETRAINED" \
  --out "$OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"

echo "[Eval] Results saved to: $OUT_DIR/eval_${SPLIT}_${FEWSHOT}shot.jsonl"
