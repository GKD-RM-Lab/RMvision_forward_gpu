#include "vino_inference.hpp"


/*加载并且初始化模型*/
void YoloInferencd_vino::load(cv::String model_path, cv::String bin_path)
{
    //载入模型
    std::shared_ptr<ov::Model> model = core.read_model(model_path, bin_path);
    //编译模型到设备
    compiled_model = core.compile_model(model, "GPU");
    //创建推理请求
    infer_request = compiled_model.create_infer_request();

    //获取输入类型
    input_port = compiled_model.input();
    input_type = input_port.get_element_type();
    input_shape = input_port.get_shape(); 

    batch = input_shape[0];
    channels = input_shape[1];
    net_height = input_shape[2];
    net_width  = input_shape[3];

    //debug
    std::cout << input_port << '\n' << input_type << "\n" << input_shape << std::endl;
}

/*图像前处理*/
//todo: GPU化
ov::Tensor YoloInferencd_vino::post_process(cv::Mat inputImage)
{
    cv::Mat resized_image;
    //resize
    cv::resize(inputImage, resized_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
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
    //最后获得的输入张量
    ov::Tensor input_tensor = ov::Tensor(input_type, {batch, channels, net_height, net_width}, input_data.data());
    return input_tensor;
}

void YoloInferencd_vino::forward(ov::Tensor input)
{
    infer_request.set_input_tensor(input);

    //执行推理
    infer_request.infer();
    //获取输出
    ov::Output<const ov::Node> output_port = compiled_model.output();
    ov::Tensor output_tensor = infer_request.get_output_tensor(0);

    //查看输出
    float* output_data = output_tensor.data<float>();
    size_t num_boxes = output_tensor.get_shape()[1];
    size_t stride    = output_tensor.get_shape()[2];
    std::cout << output_tensor.get_shape() << std::endl;


}

YoloInferencd_vino::YoloInferencd_vino()
{
}

YoloInferencd_vino::~YoloInferencd_vino()
{
}
