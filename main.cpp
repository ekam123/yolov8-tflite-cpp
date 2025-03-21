#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "yolov8.h"

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>


using namespace std;
using namespace cv;


int main(int argc, char **argv)
{

    std::string file_path = "../models/yolov8n_float32.tflite";
    
    Mat frame = cv::imread("../assets/bus.jpg");
    Mat img = frame.clone();

    TF_Inference inference = TF_Inference(file_path);
    inference.performInference(img);

    cv::imshow("Detections", img);
    cv::waitKey(0);

    return -1;
}