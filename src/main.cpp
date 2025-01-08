#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

// YOLOv8 默认输入尺寸为 640
static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;

// YOLOv8 官方导出 onnx 后，通常输出大小为 [batch, num_detection, 85]，
// 其中 85 = 4 (bbox) + 1 (obj conf) + 80 (类别概率)，以 COCO80 类为例。
// 如果你用的是自定义数据集，类别数可能不一样，需要自行适配。
static const int OUTPUT_DIM = 85;

// 置信度阈值
static const float CONF_THRESHOLD = 0.4f;
// NMS 阈值
static const float NMS_THRESHOLD = 0.45f;

struct Detection {
    cv::Rect box;   // bounding box
    float conf;     // confidence
    int classId;    // class index
};

void parseYoloOutput(
    cv::Mat& output,           // 输出张量，形状 [1, 84, 8400]
    cv::Mat& frame,            // 原始图像
    std::vector<Detection>& detections,
    float confThreshold  // 置信度阈值
);

void gpu_accel_check();

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <onnx_model_path> <image_path>" << std::endl;
        return -1;
    }

    gpu_accel_check();

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    // 1. 读取模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // 先使用PCU：
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 2. 读取测试图像
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    // 3. 准备输入张量 (letterbox 或者直接 resize 到 640x640)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

    // BGR -> RGB (看你的模型是否需要，如果导出的模型在 onnx 内部做了 BGR->RGB，就可以省略)
    // cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 转为 blob，注意缩放因子 / 均值 / 方差，看模型导出时 ultralytics 默认是否做了归一化
    // ultralytics 通常使用 [0,1] 归一化，无额外均值方差
    // 参考： https://docs.ultralytics.com/usage/cfg/
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0,
                                          cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                                          cv::Scalar(0,0,0),
                                          true, false);

    // 4. 前向推理
    net.setInput(blob);
    cv::Mat output = net.forward();

    // output shape: [1, num_outputs, 85]
    // num_outputs 的值可能根据图像而变(动态尺寸可以导致 Anchor Free)，
    // 这里假定导出的是固定 640x640 的输入，输出形状一般是 [1, 8400, 85] 之类
    std::cout << "Output shape: " << output.size << std::endl;

    // 5. 解析输出
    std::vector<Detection> detections;
    // float* data = (float*)output.data;
    //debug
    for(int i=0; i<83; i++){
        std::cout << output.at<float>(0, 2, i) << ",";
    }
    std::cout << std::endl;

    parseYoloOutput(output, frame, detections, 0.5);

    std::cout << detections.size() << std::endl;

    // 6. 执行 NMS
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (auto &d : detections) {
        boxes.push_back(d.box);
        confidences.push_back(d.conf);
    }
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    // 7. 可视化结果
    for (auto i : indices) {
        Detection det = detections[i];
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("ID: %d Conf: %.2f", det.classId, det.conf);
        cv::putText(frame, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }

    // 8. 显示或保存结果
    cv::imwrite("result.jpg", frame);
    std::cout << "Result saved to result.jpg" << std::endl;

    return 0;
}

/*解析YOLO输出*/
void parseYoloOutput(
    cv::Mat& output,           // 输出张量，形状 [1, 84, 8400]
    cv::Mat& frame,            // 原始图像
    std::vector<Detection>& detections,
    float confThreshold = 0.5  // 置信度阈值
)
{
    // 假设输出张量的三个维度分别是：batch = 1, channels = 84, numPred = 8400
    // Mat 的 dims = 3, size[0] = 1, size[1] = 84, size[2] = 8400
    const int channels = output.size[1];  // 84
    const int numPred  = output.size[2];  // 8400

    // 将输出强制转换为 float 指针
    float* data = (float*)output.data;

    // 通道布局： [1, channel=84, anchor=8400]
    // 索引计算方式： data[ 通道 * numPred + anchor_idx ]
    // 其中 anchor_idx 从 0 ~ (numPred-1)

    for (int i = 0; i < numPred; ++i) {
        // 取出预测框相关数值
        float xCenter   = data[0 * numPred + i];  // 第 0 个通道
        float yCenter   = data[1 * numPred + i];  // 第 1 个通道
        float w         = data[2 * numPred + i];  // 第 2 个通道
        float h         = data[3 * numPred + i];  // 第 3 个通道
        float box_conf  = data[4 * numPred + i];  // 第 4 个通道 (obj_conf)

        if (box_conf < confThreshold) {
            // 如果目标置信度过低，跳过
            continue;
        }

        // 找到最大类别置信度及其类别 ID
        float maxClassScore = -1.0f;
        int   maxClassId    = -1;

        for (int c = 5; c < channels; ++c) {
            float classScore = data[c * numPred + i];
            if (classScore > maxClassScore) {
                maxClassScore = classScore;
                maxClassId    = c - 5;  // 类别下标从 0 开始
            }
        }

        // 计算总置信度 = box_conf * maxClassScore
        float confidence = box_conf * maxClassScore;
        if (confidence < confThreshold) {
            // 如果最终置信度过低，跳过
            continue;
        }

        // 将归一化的 xywh 解码成在原图上的框
        int centerX = static_cast<int>(xCenter * frame.cols);
        int centerY = static_cast<int>(yCenter * frame.rows);
        int width   = static_cast<int>(w * frame.cols);
        int height  = static_cast<int>(h * frame.rows);

        int left = centerX - width  / 2;
        int top  = centerY - height / 2;

        // 存储检测结果
        Detection det;
        det.box      = cv::Rect(left, top, width, height);
        det.conf     = confidence;
        det.classId  = maxClassId;
        detections.push_back(det);
    }
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
}