#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

APP_DIR="$ROOT/examples/qwen_lora_finetune"
BUILD_DIR="$APP_DIR/build-android"
NDK_ROOT="$(mf_android_resolve_ndk)"
TOOLCHAIN="$NDK_ROOT/build/cmake/android.toolchain.cmake"

if [ ! -f "$TOOLCHAIN" ]; then
  echo "找不到 Android NDK toolchain: $TOOLCHAIN" >&2
  exit 1
fi

cmake -S "$APP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --target train_wikitext -j

echo "[Android Build] Binary: $BUILD_DIR/train_wikitext"
