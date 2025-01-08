# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.0)

# 设置你的项目名称
project(MyOpenCVProject)

# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 打印 OpenCV 版本
message(STATUS "Using OpenCV version ${OpenCV_VERSION}")

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# 将 /src 目录下的所有源文件添加到变量 SOURCE_FILES 中
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# 指定生成目标
add_executable(cl_yolo ${SOURCE_FILES})

# 添加链接库
target_link_libraries(cl_yolo ${OpenCV_LIBS})
