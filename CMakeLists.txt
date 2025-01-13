cmake_minimum_required(VERSION 3.0)
project(cl_yolo)

# 寻找 OpenCV 包
find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV version ${OpenCV_VERSION}")

# 设置 OpenVINO Developer Package 的目录
set(OpenVINODeveloperPackage_DIR "/usr/lib/cmake/openvino2024.6.0")

# 寻找 OpenVINO 包和 OpenVINO Developer Package
find_package(OpenVINO REQUIRED)
# find_package(OpenVINODeveloperPackage REQUIRED)

message(STATUS "Using OpenVINO version ${openvino_VERSION}")
# set(OpenVINO_FOUND True)


# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(${OpenVINO_INCLUDE_DIRS})  # 确保包含 OpenVINO 的头文件

# 所有源文件
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# 创建可执行文件
add_executable(cl_yolo ${SOURCE_FILES})

# 链接库
target_link_libraries(cl_yolo ${OpenCV_LIBS} openvino::runtime)

# 为目标添加 OpenVINO 特定编译器标志
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(cl_yolo PRIVATE -Wall)
endif()

option(ENABLE_TEMPLATE_REGISTRATION "Enables registration of TEMPLATE plugin" OFF)
if(ENABLE_TEMPLATE_REGISTRATION)
    target_compile_definitions(cl_yolo PRIVATE ENABLE_TEMPLATE_REGISTRATION)
endif()
