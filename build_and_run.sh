#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BUILD_DIR="build"
BUILD_TYPE="Rcelease"
GENERATOR="Ninja"     # or "Unix Makefiles" if you prefer
RUN_TARGET="run_llm"  # matches the executable name in CMakeLists.txt

# ------------------------------------------------------------
# Detect platform (macOS vs iOS cross-build)
# ------------------------------------------------------------
OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || echo "")
IOS_SYSROOT=$(xcrun --sdk iphoneos --show-sdk-path 2>/dev/null || echo "")

# For iOS cross-compile, uncomment:
# CMAKE_PLATFORM_FLAGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=$IOS_SYSROOT -DCMAKE_OSX_ARCHITECTURES=arm64"

# For macOS native:
CMAKE_PLATFORM_FLAGS="-DCMAKE_OSX_SYSROOT=$OSX_SYSROOT"

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
echo "[+] Configuring CMake..."
cmake -S . -B "$BUILD_DIR" \
  -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  $CMAKE_PLATFORM_FLAGS

echo "[+] Building project..."
cmake --build "$BUILD_DIR" --target "$RUN_TARGET" -j$(sysctl -n hw.ncpu)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
echo "[+] Running executable..."
"$BUILD_DIR/$RUN_TARGET"
