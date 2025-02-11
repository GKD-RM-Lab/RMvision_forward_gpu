#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>



#include "rmyolov7_inference.h"
#include "HIKdriver.hpp"

#include "timer.hpp"

//cam calib
#include "cam_calibration.hpp"

//pnpsolver
#include "PNPsolver.hpp"

//openvino
#include <openvino/openvino.hpp>

#include <thread>

std::string label2string(int num);
cv::Mat visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result);
void removePointsOutOfRect(std::vector<cv::Point2f>& kpt, const cv::Rect2f& rect);
int findMissingCorner(const std::vector<cv::Point2f>& pts);

void gpu_accel_check();

int main(int argc, char** argv) {

    /*相机读取线程*/
    std::thread cameraThread(HIKcamtask);
    cv::Mat inputImage;

    /*相机标定*/
    if(argc > 1)
    {
        std::string command_str;
        command_str = argv[1];    
        std::cout << command_str << std::endl;
        if(command_str == "--calibration")
        {
            //相机标定
            std::cout << "into camera calibration ....." << std::endl;
            calibration_main();
            return 0;
        }else{
            std::cout << "usage: " << std::endl;
            std::cout << "--calibration -> to calibrate camera" << std::endl;
        }
    }


    
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

        inputImage = rect_cut(inputImage);

        /*识别*/
        //识别图像（前处理+推理+后处理）
        timer.begin();
        result = model.work(inputImage);
        timer.end();
        std::cout << "total time:" << timer.read() << std::endl;
        std::cout << "--------------------" << std::endl;
        
        /*PNP*/
        //角点预处理
        for(size_t j=0; j<result.size(); j++)
        {
            //剔除无效点
            removePointsOutOfRect(result[j].kpt, result[j].rect);

            //四点都有=有解
            if(result[j].kpt.size() == 4)
            {
                result[j].pnp_is_calculated = 0;
            }

            //四缺一的情况下，确定缺了哪个角点
            if(result[j].kpt.size() == 3)
            {
                result[j].kpt_lost_index = findMissingCorner(result[j].kpt);
                result[j].pnp_is_calculated = 0;
            }
            
            //有效角点小于三判定pnp无解
            if(result[j].kpt.size() < 3)
            {
                result[j].pnp_is_calculated = -1;   
            }

        }

        //pnp求解
        pnp.calculate_all(result);

        // //debug
        // if(result.size() > 0){
        //     std::cout << result[0].pnp_tvec << std::endl;
        // }


        //fps
        char text[50];
        std::sprintf(text, "%.2fps, %.2fms", 1000/inf_time, inf_time);
        cv::putText(inputImage, text, cv::Point(0,30)
            , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,255,0), 3);
        
        //输出信息&绘图
        inputImage = visual_label(inputImage, result);
        cv::imshow("label", inputImage);
        if(cv::waitKey(1) == 'q') break;

        timer2.end();
        std::cout << "display->" << 1000/timer2.read() << "fps" << std::endl;
        timer2.begin();

    }
    return 0;
}

//角点四缺一时候，用来判断缺了哪一个角点
//返回值：0-左上，1-左下，2-右下，3-右上
//模型返回角点的顺序：左上->左下->右下->右上
int findMissingCorner(const std::vector<cv::Point2f>& trianglePoints)
{
    if (trianglePoints.size() != 3)
        return -1;  

    // 计算三条边长度
    double d01 = cv::norm(trianglePoints[0] - trianglePoints[1]);
    double d12 = cv::norm(trianglePoints[1] - trianglePoints[2]);
    double d20 = cv::norm(trianglePoints[2] - trianglePoints[0]);

    // 找出最长的边
    int gapIndex = 0;
    double maxGap = d01;
    if (d12 > maxGap) { maxGap = d12; gapIndex = 1; }
    if (d20 > maxGap) { maxGap = d20; gapIndex = 2; }

    // 判断缺失角
    if (gapIndex == 0)
    {
        return 1;
    }
    else if (gapIndex == 1)
    {
        return 2;
    }
    else  
    {
        if (d01 < d12)
            return 3;
        else
            return 0;
    }
}

//label -> 标签字符串
std::string label2string(int num) {
    std::vector<std::string> class_names = {
        "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };
    return class_names[num];
}

//可视化results
cv::Mat visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result)
{
    if(result.size() > 0)
    {
        for(size_t j=0; j<result.size(); j++)
        {
            //画出所有有效点
            for(size_t i=0; i<result[j].kpt.size(); i++)
            {
                cv::circle(inputImage, result[j].kpt[i], 3, cv::Scalar(0,255,0), 3);
                char text[10];
                std::sprintf(text, "%ld", i);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[i].x, result[j].kpt[i].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 2);
            }

            //判定框
            cv::rectangle(inputImage, result[j].rect, cv::Scalar(255,0,0), 5);

            if(result[j].kpt.size() == 4)
            {
                cv::line(inputImage, result[j].kpt[0], result[j].kpt[1], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[1], result[j].kpt[2], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[2], result[j].kpt[3], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[3], result[j].kpt[0], cv::Scalar(0,255,0), 5);
                char text[50];
                std::sprintf(text, "%s - P%.2f", label2string(result[j].label).c_str(), result[j].prob);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[3].x, result[j].kpt[3].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 3);
                //pnp结果
                if(result[j].pnp_is_calculated == 1)
                {
                    char text[50];
                    std::cout << result[j].pnp_tvec << std::endl;
                    std::sprintf(text, "x%.2fy%.2fz%.2f", result[j].pnp_tvec.at<double>(0)
                    , result[j].pnp_tvec.at<double>(1), result[j].pnp_tvec.at<double>(2));
                    cv::putText(inputImage, text, cv::Point(result[j].kpt[3].x + 10, result[j].kpt[3].y + 30)
                    , cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 3);
                }
            }

            if(result[j].kpt.size() == 3)
            {
                cv::line(inputImage, result[j].kpt[0], result[j].kpt[1], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[1], result[j].kpt[2], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[2], result[j].kpt[0], cv::Scalar(0,255,0), 5);
                char text[50];
                std::sprintf(text, "%s - %d", label2string(result[j].label).c_str(), result[j].kpt_lost_index);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[2].x, result[j].kpt[2].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 3);
                //pnp结果
                if(result[j].pnp_is_calculated == 1)
                {
                    char text[50];
                    std::cout << result[j].pnp_tvec << std::endl;
                    std::sprintf(text, "x%.2fy%.2fz%.2f", result[j].pnp_tvec.at<double>(0)
                    , result[j].pnp_tvec.at<double>(1), result[j].pnp_tvec.at<double>(2));
                    cv::putText(inputImage, text, cv::Point(result[j].kpt[2].x + 10, result[j].kpt[2].y + 30)
                    , cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 3);
                }
            }

        }
    }
    return inputImage;
}

/*剔除不在判定框中的特征点*/
void removePointsOutOfRect(std::vector<cv::Point2f>& kpt, const cv::Rect2f& rect)
{
    // 使用 remove_if + erase 在原地剔除不在矩形内的点
    kpt.erase(
        std::remove_if(kpt.begin(), kpt.end(),
            [&rect](const cv::Point2f& p) {
                // 若点不在矩形内，返回 true 表示要被移除
                return !rect.contains(p);
            }
        ),
        kpt.end()
    );
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
