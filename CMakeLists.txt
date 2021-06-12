cmake_minimum_required(VERSION 3.19)

project(IRGPU LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")

include_directories("src/" ${OpenCV_INCLUDE_DIRS})

set(SRC_CPU
    "src/load_image.cc"
)

set(SRC_GPU
    "test.cu"
    "test.cuh"
)

add_executable(cpu-lbp-engine)
target_sources(cpu-lbp-engine PRIVATE "${SRC_CPU}")

add_library(gpu-lbp-engine STATIC ${SRC_GPU})
target_compile_features(gpu-lbp-engine PUBLIC cxx_std_20)
set_target_properties(gpu-lbp-engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cpu-lbp-engine
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    gpu-lbp-engine)

