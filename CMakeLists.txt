cmake_minimum_required(VERSION 3.19)

project(IRGPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")

include_directories("src/" ${OpenCV_INCLUDE_DIRS})

set(SRC_CPU
  "src/load_image.cc"
)

find_package(OpenCV REQUIRED)

add_executable(cpu-lbp-engine)
target_sources(cpu-lbp-engine PRIVATE "${SRC_CPU}")
target_link_libraries(cpu-lbp-engine
	#	${OpenCV_LIBS}
	"opencv_core;opencv_imgcodecs;opencv_highgui")

