cmake_minimum_required(VERSION 3.19)

project(IRGPU LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)

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
    "src/misc/load_kmeans.cc"
    "src/misc/load_kmeans.hh"
    "src/misc/build_lut.hh"
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

