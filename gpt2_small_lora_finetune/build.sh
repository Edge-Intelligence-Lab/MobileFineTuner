#!/bin/zsh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$DIR/build"
mkdir -p "$BUILD_DIR"
cmake -S "$DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
cmake --build "$BUILD_DIR" -j
echo "[Build] Done. Binaries at: $BUILD_DIR"


