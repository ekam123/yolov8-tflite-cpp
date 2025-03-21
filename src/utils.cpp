#include "utils.h"
#include <fstream>
#include <cstdlib>  // For rand() and srand()
#include <ctime> 
#include <opencv2/opencv.hpp>

using namespace std;

void YOLO_UTILS::printImageValues(cv::Mat &arr, int rows, int cols)
{
    int channels = arr.channels();
    cout << "Channels: " << channels << endl;
    for (int i = 0; i < rows; i++) {
        uchar* ptr = arr.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            cout << "[";
            cout << (int)ptr[j * channels] << " ,";
            cout << (int)ptr[j * channels + 1] << " ,";
            cout << (int)ptr[j * channels + 2] << endl;
            cout << "] ";
        }
        cout << endl;
    }
}

void YOLO_UTILS::printImageValuesFloat(cv::Mat& arr, int rows, int cols) 
{
    int channels = arr.channels();
    cout << "Channels: " << channels << endl;
    for (int i = 0; i < rows; i++) {
        float* ptr = arr.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            cout << "[";
            cout << ptr[j * channels] << " ,";
            cout << ptr[j * channels + 1] << " ,";
            cout << ptr[j * channels + 2] << endl;
            cout << "] ";
        }
        cout << endl;
    }
}

std::vector<std::string> YOLO_UTILS::loadClasses(const std::string &filename) {
    std::vector<std::string> classes;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return classes;
    }
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classes.push_back(line);
        }
    }
    file.close();
    return classes;
}

std::vector<cv::Scalar> YOLO_UTILS::generateColorPalette(int num_classes) {
    std::vector<cv::Scalar> color_palette;
    std::srand(std::time(nullptr)); 

    for (int i = 0; i < num_classes; ++i) {
        int hue = std::rand() % 180;  
        int sat = 150 + std::rand() % 106;  
        int val = 150 + std::rand() % 106;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
        color_palette.emplace_back(color[0], color[1], color[2]);
    }
    return color_palette;
}

void YOLO_UTILS::drawDetections(cv::Mat &img, const cv::Rect &box, float score, int class_id, 
                                const std::vector<cv::Scalar> &color_palette, const std::vector<std::string> &classes) {
    cv::Scalar color = color_palette[class_id];
    cv::rectangle(img, box, color, 2);
    std::string label = classes[class_id] + ": " + cv::format("%.2f", score);
    int baseline = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int label_x = box.x;
    int label_y = box.y - 10;
    if (label_y < label_size.height) {
        label_y = box.y + box.height + label_size.height;
    }
    cv::rectangle(img, cv::Point(label_x, label_y - label_size.height),
                cv::Point(label_x + label_size.width, label_y + baseline), color, cv::FILLED);
    cv::putText(img, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}