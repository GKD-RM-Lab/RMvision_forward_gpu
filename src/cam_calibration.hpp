#ifndef CALIB_H
#define CALIB_H

#define IMG_COUNT 60
#define SAMPLE_PERIOD 40


#include <iostream>

// opencv
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>

// cam_driver
#include "HIKdriver.hpp"

typedef struct
{
    cv::UMat frame;
    cv::UMat frame_gray;
    std::vector<cv::UMat> frame_calib;
    cv::VideoCapture cam;
    int sample_period_count = 0;
    int cam_id = 0;             //相机ID
    long int plate_lasttime = 0;    //上次检测到标定板的时间
    int image_sample_isok = 0;
    std::string video_path;     //或，视频文件路径
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point3f> objP;
}camera_cali_type;

class camrea_calibtation
{
private:
    /* data */
public:
    camera_cali_type camrea;
    cv::Mat mtx, dist;
    std::vector<cv::Mat> rvecs, tvecs;

    /*************** calibration settings ****************/
    cv::Size boardSize; // 棋盘格的尺寸
    float squareSize = 20.0f; // 每格的大小，单位mm
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> corners;
    // 创建棋盘格的3D点
    std::vector<cv::Point3f> objP;
    
    void init(){
        /*************** cam settings ****************/
        camrea.cam.open(camrea.cam_id, cv::CAP_V4L2);
        camrea.cam.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        camrea.cam.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        camrea.cam.set(cv::CAP_PROP_FPS, 60);
        camrea.cam.set(cv::CAP_PROP_TEMPERATURE, 5000);
        // std::cout << cam.get(cv::CAP_PROP_EXPOSURE) << std::endl;
        camrea.cam.set(cv::CAP_PROP_EXPOSURE, 100);
        camrea.cam.read(camrea.frame);
        boardSize.height = 9;
        boardSize.width = 7;

        // 创建棋盘格的3D点
        std::vector<cv::Point3f> objP;
        for (int i = 0; i < boardSize.height; ++i) {
            for (int j = 0; j < boardSize.width; ++j) {
                objP.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
            }
        }
    }

    int image_add(cv::Mat frame){

       //get calibration corners
        cv::cvtColor(camrea.frame, camrea.frame_gray, cv::COLOR_BGR2GRAY);

        if(!cv::findChessboardCorners(camrea.frame_gray, boardSize, corners)){
            return -1;
        }
        //draw corners
        cv::drawChessboardCorners(camrea.frame, boardSize, corners, true);

        /*Get sample images for calibration*/
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
                return 0;
            }
        }

    }

    camrea_calibtation(/* args */);
    ~camrea_calibtation();
};




extern int calibration_main();

#endif