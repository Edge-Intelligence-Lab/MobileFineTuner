#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/lib/asset_paths.sh"
source "$ROOT/scripts/android/android_env.sh"

DEVICE_ROOT="${DEVICE_ROOT:-/sdcard/MobileFineTuner}"
QNLI_JSONL_DIR="${QNLI_JSONL_DIR:-$ROOT/data/qnli/qwen_qnli_s64}"
PUSH_MODEL="${PUSH_MODEL:-0}"
MODEL_DIR=""
if [ "$PUSH_MODEL" = "1" ]; then
  MODEL_DIR="${QWEN_MODEL_DIR:-$(mf_resolve_model_dir qwen "$ROOT/qwen_lora_finetune/pretrained")}"
fi

ADB_BIN="$(mf_android_resolve_adb)"

"$ADB_BIN" shell "mkdir -p '$DEVICE_ROOT/models' '$DEVICE_ROOT/data/qnli'"
"$ADB_BIN" push "$QNLI_JSONL_DIR" "$DEVICE_ROOT/data/qnli/" >/dev/null

if [ "$PUSH_MODEL" = "1" ]; then
  "$ADB_BIN" push "$MODEL_DIR" "$DEVICE_ROOT/models/" >/dev/null
fi

echo "[Stage] device_root=$DEVICE_ROOT"
echo "[Stage] qnli_jsonl=$DEVICE_ROOT/data/qnli/$(basename "$QNLI_JSONL_DIR")"
if [ "$PUSH_MODEL" = "1" ]; then
  echo "[Stage] model=$DEVICE_ROOT/models/$(basename "$MODEL_DIR")"
else
  echo "[Stage] model push skipped; expecting model already on device"
fi
