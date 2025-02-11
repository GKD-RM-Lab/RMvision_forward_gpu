#include "opencv2/opencv.hpp"
#include "rmyolov7_inference.h"

class pnp_solver
{
private:
    std::vector<cv::Point3f> object_4Points;      //角点 - 物体坐标系(四点都有)
    std::vector<std::vector<cv::Point3f>> object_3Points;      //角点 - 物体坐标系（缺一点）
    // std::vector<cv::Point2f> imagePoints;       //角点 - 图像坐标系
public:
    /*参数*/
    cv::Mat cameraMatrix;   //相机矩阵
    cv::Mat distCoeffs;     //畸变向量
    float squareSize = 55;  //正方形边长，单位：mm
    /*函数*/
    int calculate_single(yolo_kpt::Object &armor);
    int calculate_all(std::vector<yolo_kpt::Object> &armors);
    pnp_solver(const std::string& filename);
    void readCameraParametersFromYaml(const std::string& filename);
    ~pnp_solver();
};