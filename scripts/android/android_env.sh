#!/usr/bin/env bash

mf_android_host_tag() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) echo "darwin-arm64" ;;
    Darwin-*) echo "darwin-x86_64" ;;
    Linux-aarch64|Linux-arm64) echo "linux-aarch64" ;;
    Linux-*) echo "linux-x86_64" ;;
    *)
      echo "Unsupported host for Android NDK: $(uname -s)-$(uname -m)" >&2
      return 1
      ;;
  esac
}

mf_android_resolve_ndk() {
  local candidate
  for candidate in "${ANDROID_NDK_ROOT:-}" "${ANDROID_NDK_HOME:-}"; do
    if [ -n "$candidate" ] && [ -f "$candidate/build/cmake/android.toolchain.cmake" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local sdk
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk" "$HOME/Android/Sdk"; do
    [ -n "$sdk" ] || continue
    if [ -d "$sdk/ndk" ]; then
      candidate="$(find "$sdk/ndk" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)"
      if [ -n "$candidate" ] && [ -f "$candidate/build/cmake/android.toolchain.cmake" ]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  echo "Android NDK not found. Set ANDROID_NDK_ROOT or ANDROID_NDK_HOME." >&2
  return 1
}

mf_android_resolve_llvm_prebuilt() {
  local ndk_root="$1"
  local host_tag
  host_tag="$(mf_android_host_tag)"

  local candidate="$ndk_root/toolchains/llvm/prebuilt/$host_tag"
  if [ -d "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  local fallback
  for fallback in "$ndk_root"/toolchains/llvm/prebuilt/*; do
    if [ -d "$fallback/bin" ]; then
      printf '%s\n' "$fallback"
      return 0
    fi
  done

  echo "Android LLVM prebuilt toolchain not found under: $ndk_root/toolchains/llvm/prebuilt" >&2
  return 1
}

mf_android_resolve_adb() {
  if [ -n "${ADB:-}" ] && command -v "$ADB" >/dev/null 2>&1; then
    printf '%s\n' "$ADB"
    return 0
  fi
  if command -v adb >/dev/null 2>&1; then
    command -v adb
    return 0
  fi

  local sdk candidate
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "$HOME/Library/Android/sdk" "$HOME/Android/Sdk"; do
    candidate="$sdk/platform-tools/adb"
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "adb not found. Set ADB or install Android platform-tools." >&2
  return 1
}
