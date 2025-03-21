#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <opencv2/opencv.hpp>


class TF_Inference {
    public:

        TF_Inference(const std::string &file_path, const float confidence = 0.25, const float nms = 0.45);

        void performInference(cv::Mat& image);

    private:

        std::pair<float,float> preprocess(cv::Mat& img);
        void postProcess(cv::Mat& image, cv::Mat& mat, std::pair<float,float> pad);

        std::unique_ptr<tflite::Interpreter> interpreter;
        std::unique_ptr<tflite::FlatBufferModel> model;

        uint8_t *input_8;
        float *float_32;
        int const I_WIDTH = 640;
        int const I_HEIGHT = 640; 
        float const confidence = 0.25f;
        float confidence_threshold;
        float nms_threshold;

};