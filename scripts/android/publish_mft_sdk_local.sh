#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
ANDROID_DIR="$ROOT/android-visualizer"

if [ -n "${JAVA_HOME:-}" ] && [ ! -x "$JAVA_HOME/bin/java" ]; then
  unset JAVA_HOME
fi

if [ -z "${JAVA_HOME:-}" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || /usr/libexec/java_home)"
fi

if [ -z "${ANDROID_HOME:-}" ] && [ -d "$HOME/Library/Android/sdk" ]; then
  export ANDROID_HOME="$HOME/Library/Android/sdk"
fi

cd "$ANDROID_DIR"
./gradlew :mft-sdk:publishReleasePublicationToLocalReleaseRepository "$@"

echo "[Android SDK] Maven repo: $ANDROID_DIR/mft-sdk/build/repo"
echo "[Android SDK] Dependency: com.mobilefinetuner:mobilefinetuner-android:${MFT_SDK_VERSION:-0.1.0}"
