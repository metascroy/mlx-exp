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
set(MLX_BUILD_METAL           ON  CACHE BOOL "" FORCE)  # keep ON for Apple GPU
set(MLX_BUILD_CPU             ON  CACHE BOOL "" FORCE)
# (optional) safetensors loader if you want it:
# set(MLX_BUILD_SAFETENSORS   ON  CACHE BOOL "" FORCE)

add_subdirectory(third-party/mlx)  # builds libmlx.a inside your build tree
# target_compile_features(mlx_llm PUBLIC cxx_std_20)
# set_property(TARGET mlx_llm PROPERTY CXX_EXTENSIONS ON)
# set_property(TARGET mlx_llm PROPERTY OBJCXX_STANDARD 20)
# set_property(TARGET mlx_llm PROPERTY OBJCXX_STANDARD_REQUIRED YES)
# set_property(TARGET mlx_llm PROPERTY OBJCXX_EXTENSIONS ON)

add_executable(run_llm src/main.cpp)
# If the MLX CMake exports a target named 'mlx' (typical), link it:
target_link_libraries(run_llm PRIVATE mlx)
target_include_directories(run_llm PRIVATE third-party/mlx/include)
target_include_directories(run_llm PRIVATE src/ops.hpp src/interpreter.hpp)
