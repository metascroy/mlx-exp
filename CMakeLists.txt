# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS ON)
# set(CMAKE_OBJCXX_STANDARD 17)
# set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
# set(CMAKE_OBJCXX_EXTENSIONS ON)



# # MLX options you likely want in a C++/iOS app:
# set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
# set(MLX_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
# set(MLX_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
# set(MLX_BUILD_BENCHMARKS      OFF CACHE BOOL "" FORCE)
# set(MLX_BUILD_METAL           ON  CACHE BOOL "" FORCE)  # keep ON for Apple GPU
# set(MLX_BUILD_CPU             ON  CACHE BOOL "" FORCE)
# # (optional) safetensors loader if you want it:
# # set(MLX_BUILD_SAFETENSORS   ON  CACHE BOOL "" FORCE)

# add_subdirectory(third-party/mlx)  # builds libmlx.a inside your build tree
# # target_compile_features(mlx_llm PUBLIC cxx_std_20)
# # set_property(TARGET mlx_llm PROPERTY CXX_EXTENSIONS ON)
# # set_property(TARGET mlx_llm PROPERTY OBJCXX_STANDARD 20)
# # set_property(TARGET mlx_llm PROPERTY OBJCXX_STANDARD_REQUIRED YES)
# # set_property(TARGET mlx_llm PROPERTY OBJCXX_EXTENSIONS ON)

# add_executable(run_llm src/main.cpp)
# # If the MLX CMake exports a target named 'mlx' (typical), link it:
# target_link_libraries(run_llm PRIVATE mlx)
# target_include_directories(run_llm PRIVATE third-party/mlx/include)
# target_include_directories(run_llm PRIVATE src/ops.hpp src/interpreter.hpp src/graph_build.hpp src/kv_cache.hpp)


cmake_minimum_required(VERSION 3.21)

# ---- Language / std ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS ON)

project(mlx_llm CXX OBJCXX)

# ---- MLX build options ----
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS      OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL           ON  CACHE BOOL "" FORCE)
set(MLX_BUILD_CPU             OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_DOCS            OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)



# set(MLX_BUILD_SAFETENSORS   ON  CACHE BOOL "" FORCE)

# ---- If libtorch is in a custom location, set CMAKE_PREFIX_PATH or Torch_DIR before configure ----
# Example:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# or
#   cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..

# Prefer to inherit Torch’s own flags (includes ABI setting, etc.)
find_package(Torch REQUIRED)  # Provides TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS and target Torch::Torch

# Append Torch’s recommended compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---- macOS / Apple Silicon quality-of-life flags ----
if(APPLE)
  # Ensure we build for arm64 on Apple Silicon (adjust if you need universal builds)
  if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
  endif()

  # Deployment target (adjust as needed)
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "" FORCE)
  endif()

  # Helpful rpath so the binary can find libtorch *.dylib in-tree during dev runs
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# ---- MLX ----
add_subdirectory(third-party/mlx)

# ---- Executable ----
add_executable(run_llm
  src/main.cpp
  # add other .cpp files here if you split files
)

# Your includes: add *directories*, not header files
target_include_directories(run_llm PRIVATE
  ${TORCH_INCLUDE_DIRS}
  third-party/mlx/include
  src
)

# Link MLX and Torch
# You can link either with the imported target or the variable; prefer the target when available.
if(TARGET Torch::Torch)
  target_link_libraries(run_llm PRIVATE mlx Torch::Torch)
else()
  target_link_libraries(run_llm PRIVATE mlx "${TORCH_LIBRARIES}")
endif()

# On Linux with libstdc++, Torch’s CMake already encodes the correct C++ ABI in TORCH_CXX_FLAGS.
# DO NOT set -D_GLIBCXX_USE_CXX11_ABI manually unless you know you must override it.
