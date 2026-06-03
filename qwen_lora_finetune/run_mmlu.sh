#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"
BIN="$SCRIPT_DIR/build/train_mmlu"
MODEL_DIR="${QWEN_MODEL_DIR:-$(mf_resolve_model_dir qwen "$SCRIPT_DIR/pretrained")}"
DATA_DIR="${QWEN_MMLU_DIR:-$(mf_resolve_mmlu_dir "$REPO_ROOT")}"
OUT_DIR="$SCRIPT_DIR/outputs"
LOG_DIR="$SCRIPT_DIR/logs"

SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${GRAD_ACCUM:-1}}"
MAX_STEPS="${MAX_STEPS:-150}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-${RANK:-8}}"
LORA_ALPHA="${LORA_ALPHA:-16}"
BASE_WEIGHT_STORAGE="${BASE_WEIGHT_STORAGE:-auto}"
SEED="${SEED:-42}"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
  cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build"
  cmake --build "$SCRIPT_DIR/build" --target train_mmlu -j
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "找不到模型目錄：$MODEL_DIR"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "找不到 MMLU 數據：$DATA_DIR"; exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

"$BIN" \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --base_weight_storage "$BASE_WEIGHT_STORAGE" \
  --seed "$SEED" \
  --output_dir "$OUT_DIR"
