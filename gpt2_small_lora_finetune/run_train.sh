#!/bin/zsh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PY="${PYTHON:-python3}"

JSONL="$REPO_ROOT/runs/mmlu_jsonl_gpt2_s128/train.jsonl"
PRETRAINED="$(cd "$(dirname "$0")" && pwd)/pretrained"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/outputs"
mkdir -p "$OUT_DIR"

STEPS="${STEPS:-200}"  # If >0, steps take precedence; otherwise use EPOCHS
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-2e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
EPOCHS="${EPOCHS:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

echo "[Train] jsonl=$JSONL"
echo "[Train] pretrained=$PRETRAINED"
echo "[Train] out_dir=$OUT_DIR"

# Use local C++ binary built in this directory
LOCAL_BIN="$(cd "$(dirname "$0")" && pwd)/build/train"
if [ ! -x "$LOCAL_BIN" ]; then
  echo "[Build] Building local binary..."
  "$(cd "$(dirname "$0")" && pwd)/build.sh"
fi

ARGS=( \
  --pretrained_dir "$PRETRAINED" \
  --jsonl_train "$JSONL" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --lr "$LR" \
  --rank "$RANK" \
  --alpha "$ALPHA" \
  --log_interval "$LOG_EVERY" \
  --lora_out "$OUT_DIR/lora_final.safetensors" \
)

if [ "${STEPS}" -gt 0 ]; then
  ARGS+=( --steps "$STEPS" )
else
  ARGS+=( --epochs "$EPOCHS" )
fi

"$LOCAL_BIN" "${ARGS[@]}"

echo "[Train] Done. LoRA artifact: $OUT_DIR/lora_adapter"


