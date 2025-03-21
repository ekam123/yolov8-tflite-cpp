#include <iostream>
#include <opencv2/opencv.hpp>


namespace YOLO_UTILS {

    // Print sample value up to the number of cols and rows selected
    void printImageValues(cv::Mat& arr, int rows, int cols); //uchar for int8

    void printImageValuesFloat(cv::Mat& arr, int rows, int cols);

    std::vector<std::string> loadClasses(const std::string &filename); //load file containing class names

    std::vector<cv::Scalar> generateColorPalette(int num_classes); //generate random colours for each class id

    void drawDetections(cv::Mat &img, const cv::Rect &box, float score, int class_id, 
                        const std::vector<cv::Scalar> &color_palette, const std::vector<std::string> &classes);
};