#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "rmyolov7_inference.h"
#include "HIKdriver.hpp"

#include "timer.hpp"

//pnpsolver
#include "PNPsolver.hpp"

//openvino
#include <openvino/openvino.hpp>

#include <thread>


//loader
#include "parameter_loader.hpp"
#include "cam_calibration.hpp"
#include "send_control.hpp"

void gpu_accel_check();

double getFlyDelay(
    double& yaw,
    double& pitch, 
    const double speed,
    const double target_x,
    const double target_y,
    const double target_z
) {
    yaw = atan2(target_y, target_x);
    double g = 9.8;
    double h = target_z;
    double d = sqrt(target_x * target_x + target_y * target_y);
    double t = sqrt(d * d + h * h) / speed;

    for(int i = 0; i < 5; i++) {
        
        pitch = asin((h + 0.5 * g * t * t) / (speed * t));
        
        if (std::isnan(pitch)) {
            pitch = 0.0;
        }

        t = d / (speed * cos(pitch));
    }
    return t;
}

int main(int argc, char** argv) {

    init_send("192.168.1.211");
    //载入参数
    para_load("../config/config.yaml");
    // return 0;   //debug

    /*相机读取线程*/
    std::thread cameraThread(HIKcamtask);
    cv::Mat inputImage;

    float total_time;   //debug 每识别周期时间

    //启用opencl
    cv::ocl::setUseOpenCL(true);
    gpu_accel_check();

    /*推理模型*/
    yolo_kpt model;
    std::vector<yolo_kpt::Object> result;

    /*计时器*/
    Timer timer, timer2;
    timer2.begin();

    /*PNP求解器*/
    pnp_solver pnp("../config/camera_paramets.yaml");
    std::cout << "camera intrinsics is loaded to :" << std::endl;
    std::cout << "cameraMatrix:" << std::endl;
    std::cout << pnp.cameraMatrix << std::endl;
    std::cout << "distCoeffs:" << std::endl;
    std::cout << pnp.distCoeffs << std::endl;

    while(1)
    {   
        //读取视频帧
        HIKframemtx.lock();
        HIKimage.copyTo(inputImage);
        HIKframemtx.unlock();
        if(inputImage.empty()) continue;

        if(params.rect_cut == 1){
            inputImage = rect_cut(inputImage);
        }

        /*识别*/
        //识别图像（前处理+推理+后处理）
        timer.begin();
        result = model.work(inputImage);
        timer.end();
        // std::cout << "total time:" << timer.read() << std::endl;
        // std::cout << "--------------------" << std::endl;
        
        /*PNP*/
        model.pnp_kpt_preprocess(result);
        pnp.calculate_all(result);

        inputImage = model.visual_label(inputImage, result);

        char text[50];
        std::sprintf(text, "%.2f FPS", 1000.0 / total_time);
        cv::putText(inputImage, text, cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2);

        cv::imshow("YOLOv7-Detection", inputImage);

        if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 27) break;

        cv::imwrite("../last_frame.jpg", inputImage);

        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        timer2.end();
        total_time = (float)timer2.read();
        timer2.begin();
    }

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
