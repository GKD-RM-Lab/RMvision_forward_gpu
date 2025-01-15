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
    // std::cout << input_port << '\n' << input_type << "\n" << input_shape << std::endl;
}

/*图像前处理*/
//todo: GPU化
ov::Tensor YoloInferencd_vino::pre_process(cv::Mat inputImage)
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

    //解析输出
    output_data = output_tensor.data<float>();
    num_boxes = output_tensor.get_shape()[1];
    stride    = output_tensor.get_shape()[2];
    
    //debug - 查看输出
    // std::cout << "----------" << std::endl;
    // std::cout << "mun_box:" << num_boxes << std::endl;
    // std::cout << "stride:" << stride << std::endl;
    // std::cout << "shape:" << output_tensor.get_shape() << std::endl;
    // for(int j=0; j<20; j++){
    //     for(int i=0; i<stride; i++){
    //         std::cout << output_data[j*stride + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }

}

std::vector<yolo_detec_box> YoloInferencd_vino::post_process()
{
    std::vector<yolo_detec_box> results;
    yolo_detec_box result;
 
    for(int j=0; j<num_boxes; j++){
        //安装执行度筛选判定框
        if(output_data[j*stride + 4] < yolo_cong_threshold)
            continue;

        //debug 输出通过筛选的判定框
        for(int i=0; i<stride; i++){
            std::cout << output_data[j*stride + i] << ",";
        }
        std::cout << "index = " << j << std::endl;

        //解析判定框数据
        result.x1 = output_data[j*stride + 0];
        result.y1 = output_data[j*stride + 1];
        result.x2 = output_data[j*stride + 2];
        result.y2 = output_data[j*stride + 3];
        //选择分类结果
        float max_conf_class = -std::numeric_limits<float>::infinity();
        int max_conf_class_id = -1;
        for(int i=5; i<stride; i++){
            if(output_data[j*stride + i] > max_conf_class){
                max_conf_class = output_data[j*stride + i];
                max_conf_class_id = i - 5; 
            }
        }
        result.class_result = max_conf_class_id;
        result.conf = max_conf_class;

        results.push_back(result);

        //debug 判定框结构体内容
        // std::cout << "results_num=" << results.size() << std::endl;
        // std::cout << "(" << results[0].x1 << "," << results[0].y1 << ") ->";
        // std::cout << "(" << results[0].x2 << "," << results[0].y2 << "),";
        // std::cout << "class_id=" << results[0].class_result << ", conf=" << results[0].conf << std::endl;

    }

    return results;
}

YoloInferencd_vino::YoloInferencd_vino()
{
}

YoloInferencd_vino::~YoloInferencd_vino()
{
}
