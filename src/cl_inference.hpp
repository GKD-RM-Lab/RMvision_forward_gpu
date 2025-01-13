#ifndef CL_INF_CPP
#define CL_INF_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

class YoloInferencd
{
private:
    static const int INPUT_WIDTH = 640;
    static const int INPUT_HEIGHT = 640;
    cv::dnn::Net net;       //模型对象
    std::vector<cv::Mat> model_outs;    //模型推理结果
public:
    void load(cv::String model_path);
    void forward(cv::Mat inputImage);    //推理（输入图片）


    YoloInferencd();
    ~YoloInferencd();
};

#endif