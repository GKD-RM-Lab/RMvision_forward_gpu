#include "four_points.hpp"

namespace fs = std::filesystem;


void FourPointsLoader::read_label(std::string path, FourPointsLabelType& label)
{
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "failed to load file" << std::endl;
        return ;
    }

    float data[13];
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int i = 0; i < 13; ++i) {
            if (!(iss >> data[i])) {
                std::cerr << "data decod err" << std::endl;
                return ;
            }
        }
    }

    file.close();

    //debug 输出浮点数
    // for (int i = 0; i < 13; ++i) {
    //     std::cout << "data[" << i << "] = " << data[i] << std::endl;
    // }

    //解码
    label.class_id = (int)data[0];
    label.x = data[1];
    label.y = data[2];
    label.w = data[3];
    label.h = data[4];
    label.x1 = data[5];
    label.y1 = data[6];
    label.x2 = data[7];
    label.y2 = data[8];
    label.x3 = data[9];
    label.y3 = data[10];
    label.x4 = data[11];
    label.y4 = data[12];
    label.file_path = path;

}


std::vector<fs::path> FourPointsLoader::listFilesInDirectory(const fs::path& directoryPath)
{
    std::vector<fs::path> files;
    try {
        // 检查目录是否存在
        if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
            std::cerr << "Directory does not exist." << std::endl;
            return files;
        }

        // 遍历目录
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (fs::is_regular_file(entry.status())) {
                files.push_back(entry.path());  // 将文件路径添加到 vector
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return files;
}


//读取文件夹下全部标签
std::vector<FourPointsLabelType> FourPointsLoader::read_labels(std::string folder)
{
    std::vector<fs::path> files = listFilesInDirectory(folder);
    FourPointsLabelType label;
    std::vector<FourPointsLabelType> labels;

    for(uint i=0; i < files.size(); i++)
    {
        read_label(files[i], label);
        labels.push_back(label);
    }

    return labels;
}

std::string FourPointsLoader::get_image_dir(std::string labelPath)
{
    // 寻找 "/labels/" 的位置
    std::size_t pos = labelPath.find("/labels/");
    if (pos != std::string::npos) {
        // 替换 "/labels/" 为 "/images/"
        labelPath.replace(pos, 8, "/images/");
    }

    // 更改文件扩展名从 ".txt" 到 ".jpg"
    pos = labelPath.rfind(".txt");
    if (pos != std::string::npos) {
        labelPath.replace(pos, 4, ".jpg");
    }

    return labelPath;
}

//给任意某一个图片可视化标注（debug)
void FourPointsLoader::visulize_label(FourPointsLabelType label)
{
    std::string image_path = get_image_dir(label.file_path);
    cv::Mat Image = cv::imread(image_path);
    int hight = Image.rows;
    int width = Image.cols;
    cv::circle(Image, cv::Point(width * label.x1, hight * label.y1), 3, cv::Scalar(0,0,255), 5);
    cv::circle(Image, cv::Point(width * label.x2, hight * label.y2), 3, cv::Scalar(0,0,255), 5);
    cv::circle(Image, cv::Point(width * label.x3, hight * label.y3), 3, cv::Scalar(0,0,255), 5);
    cv::circle(Image, cv::Point(width * label.x4, hight * label.y4), 3, cv::Scalar(0,0,255), 5);

    cv::rectangle(Image, cv::Point((label.x - label.w*0.5) * width, (label.y - label.h*0.5) * hight),
        cv::Point((label.x + label.w*0.5) * width, (label.y + label.h*0.5) * hight), cv::Scalar(0, 255, 0), 2);

    cv::imwrite("../dataset/debug/label_visual.jpg", Image);
}