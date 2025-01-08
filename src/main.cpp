#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

void gpu_accel_check();

int main()
{

    gpu_accel_check();

    return 0;
}


void gpu_accel_check()
{
    //check opencl
    if (cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL is supported." << std::endl;
    } else {
        std::cout << "OpenCL is not supported." << std::endl;
    }
    //check DNN
    // 获取所有可用的后端
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backendsAndTargets = cv::dnn::getAvailableBackends();

    std::cout << "Available DNN Backends and Targets:" << std::endl;
    for (auto &backendAndTarget : backendsAndTargets) {
        cv::dnn::Backend backend = backendAndTarget.first;
        cv::dnn::Target target = backendAndTarget.second;

        // 根据返回的 Backend 枚举类型打印具体名称
        switch (backend) {
            case cv::dnn::DNN_BACKEND_DEFAULT:
                std::cout << "Backend: DNN_BACKEND_DEFAULT, ";
                break;
            case cv::dnn::DNN_BACKEND_HALIDE:
                std::cout << "Backend: DNN_BACKEND_HALIDE, ";
                break;
            case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE:
                std::cout << "Backend: DNN_BACKEND_INFERENCE_ENGINE, ";
                break;
            case cv::dnn::DNN_BACKEND_OPENCV:
                std::cout << "Backend: DNN_BACKEND_OPENCV, ";
                break;
            case cv::dnn::DNN_BACKEND_VKCOM:
                std::cout << "Backend: DNN_BACKEND_VKCOM, ";
                break;
            case cv::dnn::DNN_BACKEND_CUDA:
                std::cout << "Backend: DNN_BACKEND_CUDA, ";
                break;
            default:
                std::cout << "Backend ID: " << backend << ", ";
                break;
        }

        // 根据返回的 Target 枚举类型打印具体名称
        switch (target) {
            case cv::dnn::DNN_TARGET_CPU:
                std::cout << "Target: DNN_TARGET_CPU" << std::endl;
                break;
            case cv::dnn::DNN_TARGET_OPENCL:
                std::cout << "Target: DNN_TARGET_OPENCL" << std::endl;
                break;
            case cv::dnn::DNN_TARGET_OPENCL_FP16:
                std::cout << "Target: DNN_TARGET_OPENCL_FP16" << std::endl;
                break;
            case cv::dnn::DNN_TARGET_CUDA:
                std::cout << "Target: DNN_TARGET_CUDA" << std::endl;
                break;
            case cv::dnn::DNN_TARGET_CUDA_FP16:
                std::cout << "Target: DNN_TARGET_CUDA_FP16" << std::endl;
                break;
            default:
                std::cout << "Target ID: " << target << std::endl;
                break;
        }
    }
}