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


/*letterbox resize*/
cv::Mat YoloInferencd_vino::letterboxImage(const cv::Mat& src, int output_width, int output_height) {
    cv::Mat resized_img, padded_img;
    int width = src.cols, height = src.rows;
    float scale = std::min((float)output_width / width, (float)output_height / height);
    int new_width = (int)(width * scale);
    int new_height = (int)(height * scale);
    
    // Resize the image preserving aspect ratio
    cv::resize(src, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    
    // Calculate top, bottom, left and right border sizes
    int top = (output_height - new_height) / 2;
    int bottom = output_height - new_height - top;
    int left = (output_width - new_width) / 2;
    int right = output_width - new_width - left;
    
    // Make border (this effectively adds padding around the resized image)
    cv::copyMakeBorder(resized_img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    return padded_img;
}


/*图像前处理*/
//todo: GPU化
ov::Tensor YoloInferencd_vino::pre_process(cv::Mat inputImage)
{
    //resize
    cv::resize(inputImage, resized_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    // resized_image = letterboxImage(inputImage, INPUT_WIDTH, INPUT_HEIGHT);
    cv::imwrite("/home/gkd/Opencl_vision/yolo_opencl/videos/debug_resized_image.jpg", resized_image);

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
        result.x = output_data[j*stride + 0];
        result.y = output_data[j*stride + 1];
        result.w = output_data[j*stride + 2];
        result.h = output_data[j*stride + 3];
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


cv::Mat YoloInferencd_vino::visulize(std::vector<yolo_detec_box> detecbox, cv::Mat image)
{
    if(detecbox.size() == 0){
        return image;
    }

    for(int i=0; i<detecbox.size(); i++)
    {
        cv::rectangle(image, cv::Point(detecbox[i].x - detecbox[i].w*0.5, detecbox[i].y - detecbox[i].h*0.5),
        cv::Point(detecbox[i].x + detecbox[i].w*0.5, detecbox[i].y + detecbox[i].h*0.5), cv::Scalar(0, 0, 255), 2);
    }

    return image;
}


YoloInferencd_vino::YoloInferencd_vino()
{
}

YoloInferencd_vino::~YoloInferencd_vino()
{
}
