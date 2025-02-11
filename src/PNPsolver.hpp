#include "opencv2/opencv.hpp"


class pnp_solver
{
private:
    std::vector<cv::Point3f> objectPoints;      //角点 - 物体坐标系
    // std::vector<cv::Point2f> imagePoints;       //角点 - 图像坐标系
public:
    cv::Mat cameraMatrix;   //相机矩阵
    cv::Mat distCoeffs;     //畸变向量
    float squareSize = 55;  //正方形边长，单位：mm
    int calculate(std::vector<cv::Point> imagePoints, cv::OutputArray rvec, cv::OutputArray tvec);
    pnp_solver(cv::Mat mtx, cv::Mat dist);
    ~pnp_solver();
};