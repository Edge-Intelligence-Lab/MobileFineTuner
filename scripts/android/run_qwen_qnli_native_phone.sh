#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

BUILD_DIR="$ROOT/qwen_lora_finetune/build-android"
BIN_LOCAL="$BUILD_DIR/train_wikitext"
MONITOR_SRC="$ROOT/scripts/android/proc_mem_monitor.cpp"
MONITOR_LOCAL="$BUILD_DIR/proc_mem_monitor"
RESOURCE_MONITOR_SRC="$ROOT/scripts/android/adb_resource_monitor.sh"
DEVICE_TMP="${DEVICE_TMP:-/data/local/tmp/mf_qwen_qnli}"
DEVICE_ROOT="${DEVICE_ROOT:-/sdcard/MobileFineTuner}"
DEVICE_MODEL_DIR="${DEVICE_MODEL_DIR:-$DEVICE_ROOT/models/Qwen2.5-0.5B}"
DEVICE_QNLI_DIR="${DEVICE_QNLI_DIR:-$DEVICE_ROOT/data/qnli/qwen_qnli_s64}"
DEVICE_OUT_DIR="${DEVICE_OUT_DIR:-$DEVICE_ROOT/runs/qwen_qnli_mf_native}"
DEVICE_LOG_FILE="${DEVICE_LOG_FILE:-$DEVICE_OUT_DIR/console.log}"

SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
LR="${LR:-2e-4}"
LOSS_IMPL="${LOSS_IMPL:-full_dense}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGETS="${LORA_TARGETS:-qkvo}"
SEED="${SEED:-42}"
NO_SHUFFLE="${NO_SHUFFLE:-1}"
SYNTHETIC_SMOKE="${SYNTHETIC_SMOKE:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
MEMORY_MONITOR="${MEMORY_MONITOR:-1}"
MEMORY_INTERVAL_MS="${MEMORY_INTERVAL_MS:-5}"
RESOURCE_MONITOR="${RESOURCE_MONITOR:-0}"
RESOURCE_INTERVAL_MS="${RESOURCE_INTERVAL_MS:-5}"
BASE_WEIGHT_STORAGE="${BASE_WEIGHT_STORAGE:-auto}"

TARGET_FLAG="--qv_only"
if [[ "$LORA_TARGETS" == "qkvo" ]]; then
  TARGET_FLAG="--qkvo"
fi

EXTRA_FLAGS=()
if [[ "$NO_SHUFFLE" == "1" ]]; then
  EXTRA_FLAGS+=("--no_shuffle")
fi
if [[ "$SYNTHETIC_SMOKE" == "1" ]]; then
  EXTRA_FLAGS+=("--synthetic_smoke" "--smoke_steps" "$SMOKE_STEPS")
fi

if [ ! -x "$BIN_LOCAL" ]; then
  "$ROOT/scripts/android/build_qwen_android.sh"
fi

if [[ "$MEMORY_MONITOR" == "1" ]]; then
  NDK_ROOT="$(mf_android_resolve_ndk)"
  LLVM_PREBUILT="$(mf_android_resolve_llvm_prebuilt "$NDK_ROOT")"
  CLANGXX="$LLVM_PREBUILT/bin/aarch64-linux-android29-clang++"
  if [[ ! -x "$CLANGXX" ]]; then
    echo "找不到 Android clang++: $CLANGXX" >&2
    exit 1
  fi
  if [[ ! -x "$MONITOR_LOCAL" || "$MONITOR_SRC" -nt "$MONITOR_LOCAL" ]]; then
    "$CLANGXX" -std=c++17 -O2 -fPIE -pie -static-libstdc++ "$MONITOR_SRC" -o "$MONITOR_LOCAL"
  fi
fi

ADB_BIN="$(mf_android_resolve_adb)"

"$ADB_BIN" shell "mkdir -p '$DEVICE_TMP' '$DEVICE_OUT_DIR' && rm -f '$DEVICE_OUT_DIR/step_metrics.csv' '$DEVICE_LOG_FILE' '$DEVICE_OUT_DIR/memory_samples.csv' '$DEVICE_OUT_DIR/memory_summary.json' '$DEVICE_OUT_DIR/resource_samples_5ms.csv' '$DEVICE_OUT_DIR/resource_summary_5ms.txt'"
"$ADB_BIN" push "$BIN_LOCAL" "$DEVICE_TMP/train_qnli" >/dev/null
"$ADB_BIN" shell "chmod 755 '$DEVICE_TMP/train_qnli'"
if [[ "$MEMORY_MONITOR" == "1" ]]; then
  "$ADB_BIN" push "$MONITOR_LOCAL" "$DEVICE_TMP/proc_mem_monitor" >/dev/null
  "$ADB_BIN" shell "chmod 755 '$DEVICE_TMP/proc_mem_monitor'"
fi
if [[ "$RESOURCE_MONITOR" == "1" ]]; then
  "$ADB_BIN" push "$RESOURCE_MONITOR_SRC" "$DEVICE_TMP/resource_monitor" >/dev/null
  "$ADB_BIN" shell "chmod 755 '$DEVICE_TMP/resource_monitor'"
fi

COMMON_CMD="'$DEVICE_TMP/train_qnli' \
    --model_dir '$DEVICE_MODEL_DIR' \
    --jsonl_train '$DEVICE_QNLI_DIR/train.jsonl' \
    --jsonl_valid '$DEVICE_QNLI_DIR/valid.jsonl' \
    --seq_len '$SEQ_LEN' \
    --batch_size '$BATCH_SIZE' \
    --grad_accum_steps '$GRAD_ACCUM_STEPS' \
    --max_steps '$MAX_STEPS' \
    --lr '$LR' \
    --loss_impl '$LOSS_IMPL' \
    --max_grad_norm '$MAX_GRAD_NORM' \
    --lora_r '$LORA_R' \
    --lora_alpha '$LORA_ALPHA' \
    --lora_dropout '$LORA_DROPOUT' \
    --base_weight_storage '$BASE_WEIGHT_STORAGE' \
    --seed '$SEED' \
    $TARGET_FLAG \
    ${EXTRA_FLAGS[*]} \
    --output_dir '$DEVICE_OUT_DIR'"

if [[ "$MEMORY_MONITOR" == "1" ]]; then
  "$ADB_BIN" shell "cd '$DEVICE_OUT_DIR'; : > '$(basename "$DEVICE_LOG_FILE")'; \
    echo '[MF] memory_monitor=1 interval_ms=$MEMORY_INTERVAL_MS' >> '$(basename "$DEVICE_LOG_FILE")'; \
    echo '[MF] resource_monitor=$RESOURCE_MONITOR interval_ms=$RESOURCE_INTERVAL_MS' >> '$(basename "$DEVICE_LOG_FILE")'; \
    echo '[MF] synthetic_smoke=$SYNTHETIC_SMOKE smoke_steps=$SMOKE_STEPS' >> '$(basename "$DEVICE_LOG_FILE")'; \
    $COMMON_CMD >> '$(basename "$DEVICE_LOG_FILE")' 2>&1 & \
    TRAIN_PID=\$!; \
    echo \"[MF] train_pid=\$TRAIN_PID\" | tee -a '$(basename "$DEVICE_LOG_FILE")'; \
    '$DEVICE_TMP/proc_mem_monitor' \
      --pid \"\$TRAIN_PID\" \
      --interval_ms '$MEMORY_INTERVAL_MS' \
      --csv '$DEVICE_OUT_DIR/memory_samples.csv' \
      --summary '$DEVICE_OUT_DIR/memory_summary.json' & \
    MONITOR_PID=\$!; \
    RESOURCE_PID=''; \
    if [ '$RESOURCE_MONITOR' = '1' ]; then \
      '$DEVICE_TMP/resource_monitor' \
        --pid \"\$TRAIN_PID\" \
        --interval_ms '$RESOURCE_INTERVAL_MS' \
        --csv '$DEVICE_OUT_DIR/resource_samples_5ms.csv' \
        --summary '$DEVICE_OUT_DIR/resource_summary_5ms.txt' & \
      RESOURCE_PID=\$!; \
      echo \"[MF] resource_pid=\$RESOURCE_PID\" | tee -a '$(basename "$DEVICE_LOG_FILE")'; \
    fi; \
    tail -f '$(basename "$DEVICE_LOG_FILE")' & \
    TAIL_PID=\$!; \
    wait \"\$TRAIN_PID\"; TRAIN_STATUS=\$?; \
    wait \"\$MONITOR_PID\"; MONITOR_STATUS=\$?; \
    RESOURCE_STATUS=0; \
    if [ -n \"\$RESOURCE_PID\" ]; then wait \"\$RESOURCE_PID\"; RESOURCE_STATUS=\$?; fi; \
    kill \"\$TAIL_PID\" >/dev/null 2>&1 || true; \
    wait \"\$TAIL_PID\" >/dev/null 2>&1 || true; \
    echo \"[MF] train_status=\$TRAIN_STATUS monitor_status=\$MONITOR_STATUS resource_status=\$RESOURCE_STATUS\" | tee -a '$(basename "$DEVICE_LOG_FILE")'; \
    exit \"\$TRAIN_STATUS\""
else
  "$ADB_BIN" shell "cd '$DEVICE_OUT_DIR' && $COMMON_CMD 2>&1 | tee '$(basename "$DEVICE_LOG_FILE")'"
fi
