#ifndef FOUR_POINTS_H
#define FOUR_POINTS_H

#include <iostream>
#include <fstream> // 包含对文件操作的功能
#include <sstream> // 包含字符串流
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>



namespace fs = std::filesystem;

typedef struct 
{
    int class_id;
    float x;
    float y;
    float w;
    float h;
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    std::string file_path;
}FourPointsLabelType;


/*加载&重标记4points数据集*/
class FourPointsLoader
{
private:
    /* data */
public:
    void read_label(std::string path, FourPointsLabelType& label);
    std::vector<FourPointsLabelType> read_labels(std::string folder);
    std::vector<fs::path> listFilesInDirectory(const fs::path& directoryPath);
    std::string get_image_dir(std::string labelPath);
    void visulize_label(FourPointsLabelType label);
};



#endif