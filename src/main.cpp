#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "cl_inference.hpp"
#include "vino_inference.hpp"

#include "rmyolov7_inference.h"

#include "timer.hpp"

//openvino
#include <openvino/openvino.hpp>

// YoloInferencd_cl model;
YoloInferencd_vino model;


std::string label2string(int num);
cv::Mat visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result);


void gpu_accel_check();

int main(int argc, char** argv) {
    gpu_accel_check();

    //启用opencl
    cv::ocl::setUseOpenCL(true);

    cv::Mat inputImage; // = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
    inputImage = cv::imread("../videos/red2-test-3.png");


    /*犀浦模型*/
    yolo_kpt model;
    std::vector<yolo_kpt::Object> result;

    //推理
    result = model.work(inputImage);
    inputImage = visual_label(inputImage, result);
    //输出信息&绘图



    /*视频处理：*/

    //视频读取
    cv::VideoCapture video("../videos/autoaim-test-5.mp4");

    //视频写入
    cv::VideoWriter writer("../videos/debug_autoaim_label.avi"
            , cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
            , 30
            , cv::Size(1280, 720));


    Timer timer;


    while(1)
    {   
        //读取视频帧
        video.read(inputImage);
        if(inputImage.empty()) break;

        //识别图像（前处理+推理+后处理）
        timer.begin();
        result = model.work(inputImage);
        timer.end();
        std::cout << "total time:" << timer.read() << std::endl;
        std::cout << "--------------------" << std::endl;
        
        //输出信息&绘图
        inputImage = visual_label(inputImage, result);

        //写入带标签的图片到视频
        if(inputImage.empty()) break;
        writer.write(inputImage);

    }

    video.release();
    writer.release();

    // cv::imwrite("../videos/debug_labled_image.jpg", inputImage);
    // -------------------------------------------



    return 0;
}



//label -> 标签字符串
std::string label2string(int num) {
    if (num >= 0 && num <= 4) {
        // 对应 "B1" 到 "B5"
        return "B" + std::to_string(num + 1);
    } else if (num == 5 || num == 6 || num == 7) {
        // 对应 "BO", "BS"
        return (num == 5) ? "BO" : "BS";
    } else if (num >= 8 && num <= 12) {
        // 对应 "R1" 到 "R5"
        return "R" + std::to_string(num - 7);
    } else if (num == 13) {
        // 对应 "RO", "RS"
        return (num == 13) ? "RO" : "RS";
    } else {
        return "err"; // 如果超出范围
    }
}

//可视化results
cv::Mat visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result)
{
    if(result.size() > 0)
    {
        for(int j=0; j<result.size(); j++)
        {
            for(int i=0; i<4; i++)
            {
                cv::rectangle(inputImage, result[j].rect, cv::Scalar(255,0,0), 5);
                cv::circle(inputImage, result[j].kpt[i], 3, cv::Scalar(0,255,0), 3);
            }
                char text[50];
                std::sprintf(text, "%s - P%.2f", label2string(result[j].label).c_str(), result[j].prob);
                cv::putText(inputImage, text, cv::Point(result[j].rect.x, result[j].rect.y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 3);
        }
    }
    return inputImage;
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
