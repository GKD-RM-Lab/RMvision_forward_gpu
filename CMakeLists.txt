cmake_minimum_required(VERSION 3.0)

project(MyOpenCVProject)

find_package(OpenCV REQUIRED)

message(STATUS "Using OpenCV version ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

add_executable(cl_yolo ${SOURCE_FILES})

target_link_libraries(cl_yolo ${OpenCV_LIBS})
