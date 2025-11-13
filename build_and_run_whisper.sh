#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BUILD_DIR="build"
BUILD_TYPE="Release"
GENERATOR="Ninja"     # or "Unix Makefiles" if you prefer
RUN_TARGET="run_whisper"  # executable name from CMakeLists.txt

# ------------------------------------------------------------
# Detect macOS sysroot
# ------------------------------------------------------------
OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || echo "")
CMAKE_PLATFORM_FLAGS="-DCMAKE_OSX_SYSROOT=$OSX_SYSROOT"

# ------------------------------------------------------------
# Configure & Build
# ------------------------------------------------------------
echo "[+] Configuring CMake..."
cmake -S . -B "$BUILD_DIR" \
  -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  $CMAKE_PLATFORM_FLAGS

echo "[+] Building project..."
cmake --build "$BUILD_DIR" --target "$RUN_TARGET" -j"$(sysctl -n hw.ncpu)"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
echo "[+] Running executable..."
"$BUILD_DIR/$RUN_TARGET"
