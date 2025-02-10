/*
相机标定部分
有点乱，但能用就不想整理了（
*/

#include "cam_calibration.hpp"

#include "nlohmann/json.hpp"
#include <fstream>

#include <thread>

/*相机标定参数*/
camera_cali_type camera;

//json
using json = nlohmann::json;
void writeCameraParametersToJson(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& filename);
void readCameraParametersFromJson(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
int calibration_main();
long int get_sysetm_time_ms();


/*************** calibration settings ****************/
cv::Size boardSize(9, 7); // 棋盘格的尺寸
float squareSize = 10.0f; // 每格的大小，单位mm

/*可视化线程*/
int visulization_task()
{
    cv::Mat frame;
    while(1)
    {
        
        HIKframemtx.lock();
        HIKimage.copyTo(frame);
        HIKframemtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(frame.empty()) continue;
        if(cv::waitKey(1) == 'q') break;
        
        if(get_sysetm_time_ms() - camera.plate_lasttime < 10)
        {
            cv::drawChessboardCorners(frame, boardSize, camera.corners, true);
        }

        //img count
        char text[50];
        std::sprintf(text, "image:%d/%d", IMG_COUNT, camera.frame_calib.size());
        cv::putText(frame, text, cv::Point(0,30)
            , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,255,0), 3);

        cv::imshow("cam", frame);

    }
    cv::destroyAllWindows();
    return 0;
}

int calibration_main()
{

    cv::Mat mtx, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    
    // 创建棋盘格的3D点
    std::vector<cv::Point3f> objP;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objP.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }


    //可视化线程
    std::thread visulizer(visulization_task);

    while(1)
    {
        //读取相机帧
        HIKframemtx.lock();
        HIKimage.copyTo(camera.frame);
        HIKframemtx.unlock();
        if(camera.frame.empty()) continue;
        
        //检测棋盘格角点
        cv::cvtColor(camera.frame, camera.frame_gray, cv::COLOR_BGR2GRAY);
        if(!cv::findChessboardCorners(camera.frame_gray, boardSize, camera.corners)){
            continue;
        }else{
            camera.plate_lasttime = get_sysetm_time_ms();
        }
        //绘制角点
        // cv::drawChessboardCorners(camrea.frame, boardSize, camrea.corners, true);
        // cv::imshow("cam", camrea.frame);

        /*截取足够的样本帧用于相机校准*/
        camera.sample_period_count ++;
        if(camera.sample_period_count >= SAMPLE_PERIOD){
            cv::cornerSubPix(camera.frame_gray, camera.corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            camera.imagePoints.push_back(camera.corners);
            camera.objectPoints.push_back(objP);

            //get calibration imgs
            camera.sample_period_count = 0;
            camera.frame_calib.push_back(camera.frame);
            printf("get image, idx = %ld / %d\n", camera.frame_calib.size(), IMG_COUNT);
            if(camera.frame_calib.size() >= (IMG_COUNT-1)){
                break;
            }
        }

    }

    // 执行相机标定
    cv::calibrateCamera(camera.objectPoints, camera.imagePoints, cv::Size(camera.frame_gray.rows, camera.frame_gray.cols), 
                        mtx, dist, rvecs, tvecs);

    // 输出相机参数
    writeCameraParametersToJson(mtx, dist, "../config/camera_paramets.json");
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

//获取系统时间戳
long int get_sysetm_time_ms()
{
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<long int>(duration.count());
}