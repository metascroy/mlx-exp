cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS ON)

# MLX options you likely want in a C++/iOS app:
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS      OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CPU             OFF  CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL           ON  CACHE BOOL "" FORCE)
set(MLX_BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)

# Setting to ON will minimize binary size by not including pre-built MPS kernels
# and compile them on device on first use.  It does increase start up time
# I've found it actually increases binary size
set(MLX_METAL_JIT OFF CACHE BOOL "" FORCE)

add_subdirectory(third-party/mlx)

add_executable(run_llm src/main.cpp)
target_link_libraries(run_llm PRIVATE mlx)
target_include_directories(run_llm PRIVATE third-party/mlx/include)
