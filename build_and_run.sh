#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BUILD_DIR="build"
BUILD_TYPE="Release"
GENERATOR="Ninja"     # or "Unix Makefiles" if you prefer
RUN_TARGET="run_llm"  # executable name from CMakeLists.txt

# ------------------------------------------------------------
# Detect current Conda environment and locate libtorch
# ------------------------------------------------------------
CONDA_PREFIX=$(python -c "import sys, os; print(os.environ.get('CONDA_PREFIX', sys.prefix))")
TORCH_DIR_CANDIDATE=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

# Usually libtorch CMake files live under:
#   $CONDA_PREFIX/lib/python*/site-packages/torch/share/cmake/Torch
TORCH_CMAKE_DIR=$(python - <<'PY'
import torch, os
path = os.path.join(os.path.dirname(torch.__file__), "share", "cmake", "Torch")
print(path if os.path.exists(path) else "")
PY
)

if [[ -z "$TORCH_CMAKE_DIR" ]]; then
  echo "[-] Could not find TorchConfig.cmake automatically."
  echo "    Make sure PyTorch is installed in your current Conda env."
  exit 1
fi

echo "[+] Found Torch CMake path: $TORCH_CMAKE_DIR"

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
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_DIR" \
  $CMAKE_PLATFORM_FLAGS

echo "[+] Building project..."
cmake --build "$BUILD_DIR" --target "$RUN_TARGET" -j"$(sysctl -n hw.ncpu)"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
echo "[+] Running executable..."
PROMPT_IDS=/Users/scroy/Desktop/mlx-demo/prompt_ids.txt \
MODEL_PT=/Users/scroy/Desktop/mlx-demo/model.pt \
MAX_NEW_TOKENS=64 \
"$BUILD_DIR/$RUN_TARGET"
