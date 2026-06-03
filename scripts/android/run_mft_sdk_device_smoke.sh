#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
ANDROID_DIR="$ROOT/android-visualizer"
source "$ROOT/scripts/android/android_env.sh"

if [ -n "${JAVA_HOME:-}" ] && [ ! -x "$JAVA_HOME/bin/java" ]; then
  unset JAVA_HOME
fi

if [ -z "${JAVA_HOME:-}" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || /usr/libexec/java_home)"
fi

if [ -z "${ANDROID_HOME:-}" ] && [ -d "$HOME/Library/Android/sdk" ]; then
  export ANDROID_HOME="$HOME/Library/Android/sdk"
fi

run_with_timeout() {
  local seconds="$1"
  shift

  "$@" &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$seconds" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "$pid"
}

ADB="$(mf_android_resolve_adb)"
APK="$ANDROID_DIR/sdk-sample/build/outputs/apk/debug/sdk-sample-debug.apk"
PKG="com.mobilefinetuner.sdk.sample"
ACTIVITY="$PKG/.MainActivity"

cd "$ANDROID_DIR"
./gradlew :sdk-sample:assembleDebug

"$ADB" devices
"$ADB" logcat -c

if ! run_with_timeout 120 "$ADB" install --no-streaming -r "$APK"; then
  echo "[Android SDK] APK install timed out. Unlock the phone and allow USB app installation, then retry." >&2
  exit 124
fi

"$ADB" shell am start -n "$ACTIVITY"
sleep 10

LOG_OUTPUT="$("$ADB" logcat -d -t 500)"
printf '%s\n' "$LOG_OUTPUT" | rg "MFTSdkSample|MobileFineTuner|Self-test" || true

if printf '%s\n' "$LOG_OUTPUT" | rg -q "MFTSdkSample.*Self-test passed"; then
  echo "[Android SDK] Device smoke passed"
else
  echo "[Android SDK] Device smoke did not find a self-test pass marker in logcat" >&2
  exit 1
fi
