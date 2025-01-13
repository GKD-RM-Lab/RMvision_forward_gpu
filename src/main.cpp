#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "cl_inference.hpp"

#include "timer.hpp"

//openvino
#include <openvino/openvino.hpp>


// YOLOv5 默认输入尺寸为 640
static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;

YoloInferencd model;

void gpu_accel_check();

int main(int argc, char** argv) {
    gpu_accel_check();

    //启用opencl
    cv::ocl::setUseOpenCL(true);

    cv::Mat inputImage; // = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
    inputImage = cv::imread("/home/gkd/Opencl_vision/yolo_opencl/videos/IMG_20250109_003728.jpg");
    /*Open Vino*/
    //载入模型
    ov::Core core;
     std::shared_ptr<ov::Model> model = core.read_model("/home/gkd/Opencl_vision/yolo_opencl/models/yolov5n/distilbert.xml",
                                                            "/home/gkd/Opencl_vision/yolo_opencl/models/yolov5n/distilbert.bin");

    //编译模型到设备
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");

    //创建推理请求
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    //获取输入节点信息
    ov::Output<const ov::Node> input_port = compiled_model.input();
    ov::element::Type input_type = input_port.get_element_type();
    ov::Shape input_shape = input_port.get_shape(); 
    std::cout << input_port << '\n' << input_type << "\n" << input_shape << std::endl;
    //节点信息列表
    size_t batch = input_shape[0];
    size_t channels = input_shape[1];
    size_t net_height = input_shape[2];
    size_t net_width  = input_shape[3];

    //前处理 - resize
    Timer timer;
    timer.begin();
    cv::Mat resized_image;
    cv::resize(inputImage, resized_image, cv::Size(net_width, net_height));
    //BGR -> RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    //归一化
    resized_image.convertTo(resized_image, CV_32F, 1.0f / 255.0f);
    //HWC -> CHW
    std::vector<float> input_data(net_width * net_height * channels);
    // 遍历，将 HWC 数据拷贝到 CHW buffer
    int index = 0;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < net_height; ++h) {
            for (int w = 0; w < net_width; ++w) {
                input_data[index++] = resized_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    timer.end();
    //最后得到的输入
    ov::Tensor input_tensor = ov::Tensor(input_type, {batch, channels, net_height, net_width}, input_data.data());
    std::cout << "post process : " << timer.read() << std::endl;
    //执行推理
    infer_request.set_input_tensor(input_tensor);
    for(int i=0; i<10; i++){
        timer.begin();
        infer_request.infer();
        timer.end();
        std::cout << timer.read() << std::endl;
    }

    //获取输出
    ov::Output<const ov::Node> output_port = compiled_model.output();
    ov::Tensor output_tensor = infer_request.get_output_tensor(0);


    //查看输出
    float* output_data = output_tensor.data<float>();
    size_t num_boxes = output_tensor.get_shape()[1];
    size_t stride    = output_tensor.get_shape()[2];
    std::cout << output_tensor.get_shape() << std::endl;
    for(int i=0; i<83; i++){
        std::cout << output_data[i*8400 + 2] << std::endl;      //取出某一个label
    }


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