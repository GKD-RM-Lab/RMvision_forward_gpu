#include "PNPsolver.hpp"

int pnp_solver::calculate(std::vector<cv::Point> imagePoints_cv, cv::OutputArray rvec, cv::OutputArray tvec)
{
    //cv::point -> cv::point2f
    std::vector<cv::Point2f> imagePoints;
    for (const auto& p : imagePoints_cv) {
        imagePoints.emplace_back(p.x, p.y);
    }
    return cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
}

pnp_solver::pnp_solver(cv::Mat mtx, cv::Mat dist)
{
    //初始化物体坐标系
    objectPoints = {
        {0,             0,              0},       
        {0,             squareSize,     0},       
        {squareSize,    squareSize,     0},       
        {squareSize,    0,              0}        
    };
    cameraMatrix = mtx;
    distCoeffs = dist;
}

pnp_solver::~pnp_solver()
{

}