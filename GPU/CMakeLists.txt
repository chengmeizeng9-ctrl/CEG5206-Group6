cmake_minimum_required(VERSION 3.21)
project(exact_knn_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GTX 1050 is Pascal compute capability 6.1. CUDA 12.x can target it;
# CUDA 13.x cannot perform offline compilation for Pascal.
set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures")
option(KNN_GPU_SANITIZE_HOST "Enable host address/undefined sanitizers" OFF)

find_package(CUDAToolkit REQUIRED)

add_library(gpu_knn
  src/gpu_knn.cu
  src/io.cpp
  src/verify.cpp
)
target_include_directories(gpu_knn PUBLIC include)
target_link_libraries(gpu_knn PUBLIC CUDA::cudart)
target_compile_options(gpu_knn PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--fmad=false;--ftz=false;-lineinfo;-Xcompiler=-Wall,-Wextra,-Wpedantic>
)
set_target_properties(gpu_knn PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

if(KNN_GPU_SANITIZE_HOST)
  target_compile_options(gpu_knn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>)
  target_link_options(gpu_knn PRIVATE -fsanitize=address,undefined)
endif()

add_executable(knn_gpu_bench src/bench.cpp)
target_link_libraries(knn_gpu_bench PRIVATE gpu_knn)
target_compile_options(knn_gpu_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(knn_gpu_tests tests/test_gpu.cu)
target_link_libraries(knn_gpu_tests PRIVATE gpu_knn)
target_compile_options(knn_gpu_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;-lineinfo;-Xcompiler=-Wall,-Wextra,-Wpedantic>)

enable_testing()
add_test(NAME gpu_correctness COMMAND knn_gpu_tests)
set_tests_properties(gpu_correctness PROPERTIES SKIP_RETURN_CODE 77)
