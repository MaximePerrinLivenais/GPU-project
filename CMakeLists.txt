cmake_minimum_required(VERSION 3.19)

project(IRGPU LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)

include(FetchContent)

# Benchmark
FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")

include_directories("src/" ${OpenCV_INCLUDE_DIRS})

set(SRC_CPU
    "src/main.cc"
    "src/misc/image-load.cc"
    "src/misc/image-load.hh"
    "src/misc/histo_to_file.cc"
    "src/misc/histo_to_file.hh"
)

set(SRC_GPU
    "src/lbp/lbp.cu"
    "src/lbp/lbp.cuh"
    "src/neighbors/knn.cu"
    "src/neighbors/knn.cuh"
)

add_executable(cpu-lbp-engine)
target_sources(cpu-lbp-engine PRIVATE "${SRC_CPU}")

add_library(gpu-lbp-engine SHARED ${SRC_GPU})
target_compile_features(gpu-lbp-engine PUBLIC cxx_std_17)
set_target_properties(gpu-lbp-engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cpu-lbp-engine
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    gpu-lbp-engine)



add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE benchmark)
