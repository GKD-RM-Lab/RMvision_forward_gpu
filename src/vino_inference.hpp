#ifndef VINO_INFER_HPP
#define VINO_INFER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include <openvino/openvino.hpp>


typedef struct
{
    float   x;
    float   y;
    float   w;
    float   h;
    float   conf;              //置信度
    int class_result;      //分类结果
}yolo_detec_box;



class YoloInferencd_vino
{
private:
    static const int INPUT_WIDTH = 640;
    static const int INPUT_HEIGHT = 640;

    ov::Core core;                      //ov core
    ov::CompiledModel compiled_model;   //模型编译对象
    ov::InferRequest infer_request;     //推理请求

    /*输入类型*/
    ov::Output<const ov::Node> input_port;
    ov::element::Type input_type;
    ov::Shape input_shape;

    /*输入节点信息列表*/
    size_t batch;
    size_t channels;
    size_t net_height;
    size_t net_width;

    ov::Tensor input_tensor;    //输入张量

    /*模型输出数据列表*/
    float* output_data;
    size_t num_boxes;
    size_t stride;
    float yolo_cong_threshold = 0.08;   //置信度阈值

public:
    void load(cv::String model_path, cv::String bin_path);
    ov::Tensor pre_process(cv::Mat inputImage);          //前处理
    void forward(ov::Tensor input);   
    std::vector<yolo_detec_box> post_process();
    cv::Mat visulize(std::vector<yolo_detec_box> detecbox, cv::Mat image);
    cv::Mat letterboxImage(const cv::Mat& src, int output_width, int output_height);
    YoloInferencd_vino();
    ~YoloInferencd_vino();

    cv::Mat resized_image;  //debug - 上一次被resize的图像

};



#endif