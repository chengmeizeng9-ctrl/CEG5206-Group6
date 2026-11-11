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
option(KNN_GPU_OPTIONAL_ROUTES "Build radix/scan, cuBLAS and streamed optional routes" OFF)

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

if(KNN_GPU_OPTIONAL_ROUTES)
  if(NOT TARGET CUDA::cublas)
    message(FATAL_ERROR "Optional GEMM requires cuBLAS from the CUDA 12 toolkit")
  endif()
  target_sources(gpu_knn PRIVATE src/optional_gpu_knn.cu)
  target_link_libraries(gpu_knn PUBLIC CUDA::cublas)
  add_executable(knn_optional_gpu_tests tests/test_optional_gpu.cu)
  target_link_libraries(knn_optional_gpu_tests PRIVATE gpu_knn)
  add_test(NAME optional_gpu_correctness COMMAND knn_optional_gpu_tests)
  set_tests_properties(optional_gpu_correctness PROPERTIES SKIP_RETURN_CODE 77)
  add_subdirectory(../CPU ${CMAKE_CURRENT_BINARY_DIR}/cpu_optional)
  add_executable(knn_optional_bench src/bench_optional.cpp)
  target_link_libraries(knn_optional_bench PRIVATE gpu_knn knn)
  target_compile_options(knn_optional_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()
