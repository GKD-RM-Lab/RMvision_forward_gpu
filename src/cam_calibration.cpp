/*
相机标定部分
有点乱，但能用就不想整理了（
*/

#include "cam_calibration.hpp"

#include "nlohmann/json.hpp"
#include <fstream>

camera_cali_type camrea;

//json
using json = nlohmann::json;
void writeCameraParametersToJson(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& filename);
void readCameraParametersFromJson(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
int calibration_main();

int calibration_main()
{
    /*************** calibration settings ****************/
    cv::Size boardSize(9, 7); // 棋盘格的尺寸
    float squareSize = 10.0f; // 每格的大小，单位mm
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> corners;


    cv::Mat mtx, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    
    // 创建棋盘格的3D点
    std::vector<cv::Point3f> objP;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objP.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }


    while(1)
    {
        //读取相机帧
        HIKframemtx.lock();
        HIKimage.copyTo(camrea.frame);
        HIKframemtx.unlock();
        if(camrea.frame.empty()) continue;
        
        if(cv::waitKey(1) == 'q') break;
        cv::imshow("cam", camrea.frame);

        //检测棋盘格角点
        cv::cvtColor(camrea.frame, camrea.frame_gray, cv::COLOR_BGR2GRAY);
        if(!cv::findChessboardCorners(camrea.frame_gray, boardSize, corners)){
            continue;
        }
        //绘制角点
        cv::drawChessboardCorners(camrea.frame, boardSize, corners, true);
        cv::imshow("cam", camrea.frame);

        /*截取足够的样本帧用于相机校准*/
        camrea.sample_period_count ++;
        if(camrea.sample_period_count >= SAMPLE_PERIOD){
            cv::cornerSubPix(camrea.frame_gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(objP);

            //get calibration imgs
            camrea.sample_period_count = 0;
            camrea.frame_calib.push_back(camrea.frame);
            printf("get image, idx = %ld / %d\n", camrea.frame_calib.size(), IMG_COUNT);
            if(camrea.frame_calib.size() >= (IMG_COUNT-1)){
                break;
            }
        }

    }

    // 执行相机标定
    cv::calibrateCamera(objectPoints, imagePoints, cv::Size(camrea.frame_gray.rows, camrea.frame_gray.cols), 
                        mtx, dist, rvecs, tvecs);

    // 输出相机参数
    writeCameraParametersToJson(mtx, dist, "/Users/fish/Documents/Tracker_vision/config/camera_paramets.json");
    std::cout << "Camera Matrix:\n" << mtx << std::endl;
    std::cout << "Distortion Coefficients:\n" << dist << std::endl;

    return 0;
}

// 将cameraMatrix和distCoeffs写入到JSON文件
void writeCameraParametersToJson(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& filename) {
    json j;

    for (int i = 0; i < cameraMatrix.rows; ++i) {
        for (int k = 0; k < cameraMatrix.cols; ++k) {
            j["cameraMatrix"][i][k] = cameraMatrix.at<double>(i, k);
        }
    }

    for (int i = 0; i < distCoeffs.rows; ++i) {
        for (int k = 0; k < distCoeffs.cols; ++k) {
            j["distCoeffs"][i][k] = distCoeffs.at<double>(i, k);
        }
    }

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    file << j.dump(4);  // formatted with indent 4
    if (file.fail()) {
        std::cerr << "Error writing to file: " << filename << std::endl;
    }
    file.close();
}


// 从JSON文件读取cameraMatrix和distCoeffs
void readCameraParametersFromJson(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    std::ifstream file(filename);
    json j;
    file >> j;

    // 创建相应大小的矩阵
    cameraMatrix = cv::Mat(3, 3, CV_64F);
    int distCoeffsSize = j["distCoeffs"].size() * j["distCoeffs"][0].size();
    distCoeffs = cv::Mat(distCoeffsSize, 1, CV_64F);

    // 读取cameraMatrix
    for (int i = 0; i < cameraMatrix.rows; ++i) {
        for (int k = 0; k < cameraMatrix.cols; ++k) {
            cameraMatrix.at<double>(i, k) = j["cameraMatrix"][i][k];
        }
    }

    // 读取distCoeffs
    for (int i = 0; i < distCoeffs.rows; ++i) {
        distCoeffs.at<double>(i) = j["distCoeffs"][0][i];
    }
}