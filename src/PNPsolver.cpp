#include "PNPsolver.hpp"

//pnp求解全部装甲板
int pnp_solver::calculate_all(std::vector<yolo_kpt::Object> &armors)
{
    for(int i=0; i < armors.size(); i++){
        calculate_single(armors[i]);
    }
    return 0;
}

//pnp求解单个装甲板
int pnp_solver::calculate_single(yolo_kpt::Object &armor)
{
    if(armor.pnp_is_calculated == -1)
        return -1;

    try {
        if (armor.kpt.size() == 4)
        {
            cv::solvePnP(object_4Points, armor.kpt, cameraMatrix, distCoeffs,
                         armor.pnp_rvec, armor.pnp_tvec);
            armor.pnp_is_calculated = 1;
            return 0;
        }
        else if (armor.kpt.size() == 3)
        {
            cv::solvePnP(object_3Points[armor.kpt_lost_index], armor.kpt,
                         cameraMatrix, distCoeffs, armor.pnp_rvec, armor.pnp_tvec,
                         false, cv::SOLVEPNP_SQPNP);
            armor.pnp_is_calculated = 1;
            return 0;
        }
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "solvePnP encountered an error: " << e.what() << std::endl;
        armor.pnp_is_calculated = -1;
        return -1;
    }
    
    return -1;
}

pnp_solver::pnp_solver(const std::string& filename)
{
    //装甲板参数
    float armor_width = 135;
    float light_height = 50;

    //初始化物体坐标系
    std::vector<cv::Point3f> object_all = {
        {armor_width/2,             light_height/2,              0},       //0-左上
        {armor_width/2,             -light_height/2,             0},       //1-左下
        {-armor_width/2,            -light_height/2,             0},       //2-右下
        {-armor_width/2,            light_height/2,              0}        //3-右上
    };

    object_4Points = object_all;
    for(int i=0; i<4; i++)
    {
        object_3Points.push_back(object_all);
        object_3Points[i].erase(object_3Points[i].begin() + i);
    }
    //debug
    // std::cout << object_4Points << std::endl;
    // std::cout << object_3Points[0] << std::endl;
    // std::cout << object_3Points[1] << std::endl;

    readCameraParametersFromYaml(filename);

}

// 从YAML文件读取cameraMatrix和distCoeffs
void pnp_solver::readCameraParametersFromYaml(const std::string& filename) {

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
}

pnp_solver::~pnp_solver()
{

}