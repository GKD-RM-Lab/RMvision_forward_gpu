## TODO
- 前处理&推理&后处理和socket发送 异步运行
- ~~相机标定~~
- ~~PNP解算~~
    - 区分大/小装甲板
- 把需要配置的参数提取出来放到一个yaml文件里

## 模型存在的问题
- 训练时没开旋转增强，装甲板跟相机转角过大就不能识别了
- 分类器不太准，能分清红蓝，但不太能分清编号
    - 认不出哨兵，会把XS(哨兵) 认成 X2/X3/X4/X5
    - 少数情况下会把X3/X4(步兵) 认成 XO(基地)
        - 但至少要能分出英雄跟其他，能按照大/小装甲板来PNP，也能分清红蓝
        - 鉴于联盟赛没基地，出现XO就知道他一定是步兵
        - 勉强能用

## 环境要求
- OpenCL 3.0
- Opencv >= 4.8 并且开启DNN和OpenCL模块
- Openvino 2024.6.0

## 目录结构
- model
    - 模型目录，openvino格式和onnx格式
- src
    - cl_inference 目前没用，旧模型使用opencl-DNN的推理代码
    - vino_inference 目前没用，旧模型使用openvino的推理代码
    - rmyolov7_inference 新模型的推理库
    - timer.hpp 方便测耗时的神奇小工具
    - main 

## 需要注意的参数和其他东西
- ./src/rmyolov7_inference.h 
    - DETECT_MODE 必须为0（装甲板四点）
    - CONF_THRESHOLD 置信度阈值，在有间歇性丢追或者错误识别的情况下可以调调
    - MODEL_PATH 需要与模型文件的位置对应，模型格式是openvino的格式，可以由onnx转换得到
    - IMG_SIZE 必须是640
    - `std::vector<std::string> class_names` 定义lable_id -> 标定的对应关系
- main.cpp写的有点随意，会以写死的路径从./video文件夹读输入，需要改改才能跑；里面有可视化标记数据的`visual_label`和检查opencv是否启用了gpu加速的`gpu_accel_check`

## 输出数据格式
- 每帧画面经过推理会得到一个`std::vector<yolo_kpt::Object>`的输出，每一项`yolo_kpt::Object`包含：
    - `cv::Rect_<float> rect` yolo判定框
    - `int label` 标签id
        - 红/蓝辨别是ok的，但似乎有点分不清3/4/5号步兵
    - `float prob` 置信度
    - `std::vector<cv::Point2f>kpt` 灯条四点
        - 极少数情况可能会丢一个，但三点貌似也能pnp（？
        - 几乎全部的情况下至少有三个点


## 性能测试
在GKD的老NUC上，前处理&推理&后处理一帧的用时是(单位ms)：
```
--------------------
preprocess time:6.40602
inference time:16.3719
postprocess time:0.343069
total time:23.1592
--------------------
```
GPU(intel核显)占用在62%左右，cpu会把某一个核心占用到60%左右，整体占用10%左右
直接使用的话大概可以稳定跑在40帧

可以优化的点：
- 前处理和推理用`cv::dnn::blobFromImage`或者`cv::Umat`全部跑在gpu流水线上，但这样似乎需要用openCL-kernal手写latterbox-resize(opencv好像没有能跑在GPU上的letterbox resize？)
    - 或者直接把前处理和推理异步运行
- 模型裁切到fp16精度，如果核显对fp16有优化推理速度可以快很多
    - 但是会降低四点精度

## 效果
![auto image](img/debug_labled_image.jpg)