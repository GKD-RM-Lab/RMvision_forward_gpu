#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "cl_inference.hpp"
#include "vino_inference.hpp"

#include "yolov7_kpt.h"

#include "timer.hpp"

//openvino
#include <openvino/openvino.hpp>


// YOLOv5 默认输入尺寸为 640
static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;

// YoloInferencd_cl model;
YoloInferencd_vino model;

void gpu_accel_check();

int main(int argc, char** argv) {
    gpu_accel_check();

    //启用opencl
    cv::ocl::setUseOpenCL(true);

    cv::Mat inputImage; // = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
    inputImage = cv::imread("/home/gkd/Opencl_vision/yolo_opencl/videos/IMG_20250115_204851.jpg");


    /*犀浦模型*/
    yolo_kpt model;
    std::vector<yolo_kpt::Object> result;

    result = model.work(inputImage);

    

    /*Openvino test*/
    // model.load("/home/gkd/Opencl_vision/yolo_opencl/models/yolov5-rm/distilbert.xml",
    //             "/home/gkd/Opencl_vision/yolo_opencl/models/yolov5-rm/distilbert.bin");

    // model.forward(model.pre_process(inputImage));

    // std::vector<yolo_detec_box> results;

    // results = model.post_process();

    // //debug 输出推理结果
    // std::cout << "results_num=" << results.size() << std::endl;
    // std::cout << "(" << results[0].x << "," << results[0].y << ") ->";
    // std::cout << "(" << results[0].h << "," << results[0].w << "),";
    // std::cout << "class_id=" << results[0].class_result << ", conf=" << results[0].conf << std::endl;

    // //debug 可视化
    // cv::Mat image_labeled;
    // cv::Mat resized_image = cv::imread("/home/gkd/Opencl_vision/yolo_opencl/videos/debug_resized_image.jpg");
    // image_labeled = model.visulize(results, resized_image);
    // cv::imwrite("/home/gkd/Opencl_vision/yolo_opencl/videos/debug_labled_image.jpg", image_labeled);

    /*Opencv DNN*/
    //加载
    // model.load("../models/yolov5n.onnx");
    // //推理
    // model.forward(inputImage);


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
    printf("-------------------\n");
}