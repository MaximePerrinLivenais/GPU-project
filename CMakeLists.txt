cmake_minimum_required(VERSION 3.19)

project(IRGPU LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.5.5.tar.gz
)
if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")

include_directories("src/" ${OpenCV_INCLUDE_DIRS})

set(SRC_CPU
    "src/misc/image-load.cc"
    "src/misc/image-load.hh"
    "src/misc/histo_to_file.cc"
    "src/misc/histo_to_file.hh"
    "src/misc/load_kmeans.cc"
    "src/misc/load_kmeans.hh"
    "src/misc/build_lut.hh"
    "src/pipeline/pipeline.cc"
    "src/pipeline/pipeline.hh"
)

set(SRC_GPU
    "src/lbp/lbp.cu"
    "src/lbp/lbp.cuh"
    "src/neighbors/knn.cu"
    "src/neighbors/knn.cuh"
    "src/misc/reconstruct_image.cu"
    "src/misc/reconstruct_image.cuh"
)

set(GPU_pipeline
    "src/cpu/pipeline.cc"
    "src/cpu/pipeline.hh"
    "src/cpu/lbp.hh"
    "src/cpu/lbp.cc"
    "src/cpu/knn.hh"
    "src/cpu/knn.cc"
)

add_executable(cpu-lbp-engine src/main.cc)
target_sources(cpu-lbp-engine PRIVATE "${SRC_CPU}")

add_library(gpu-lbp-engine SHARED ${SRC_GPU})
target_compile_features(gpu-lbp-engine PUBLIC cxx_std_17)
set_target_properties(gpu-lbp-engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cpu-lbp-engine
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    gpu-lbp-engine)

add_executable(bench src/bench/bench.cc)
target_sources(bench PRIVATE "${SRC_CPU}")
target_link_libraries(bench
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    gpu-lbp-engine
    cpu-test
    benchmark)

add_library(cpu-test SHARED "${GPU_pipeline}")
