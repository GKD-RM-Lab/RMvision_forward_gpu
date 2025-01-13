#include "cl_inference.hpp"

#include "timer.hpp"

YoloInferencd::YoloInferencd(/* args */)
{
}

YoloInferencd::~YoloInferencd()
{
}

void YoloInferencd::load(cv::String model_path)
{
    //加载模型
    net = cv::dnn::readNetFromONNX(model_path);
    //设置opencv后端&cpu推理
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

void YoloInferencd::forward(cv::Mat inputImage)
{
    //前处理参数
    cv::Mat blob = cv::dnn::blobFromImage(
        inputImage,               // 输入图像
        1.0,                      // 缩放因子 (scale factor)，根据需要修改
        cv::Size(640, 640),       // 模型所需的输入尺寸
        cv::Scalar(0,0,0),        // 减均值 (mean)，需要的话自行填入
        true,                     // 是否进行RGB通道顺序的交换
        false                     // 是否裁剪
    );
    net.setInput(blob);

    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
    //向前推理
    Timer timer;
    for(int i=0; i<50; i++){
        timer.begin();
        net.forward(model_outs, outNames);
        timer.end();
        std::cout << timer.read() << std::endl;
    }
    
    //查看输出向量
    std::cout << model_outs.size() << std::endl;
    std::cout << model_outs[0].size << std::endl;
    for(int i = 0; i<84; i++){
        std::cout << model_outs[0].at<float>(0,i, 1) << std::endl;
    }
}
