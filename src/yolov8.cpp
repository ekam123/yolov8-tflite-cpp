#include "yolov8.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/kernels/register.h>

using namespace std;
using namespace cv;

TF_Inference::TF_Inference(const std::string &file_path, const float confidence, const float nms)
{
    confidence_threshold = confidence;
    nms_threshold = nms;
    model = tflite::FlatBufferModel::BuildFromFile(file_path.c_str());
    if (model == nullptr)
    {
        cout << "Failed to load model. . ." << endl;
        exit(-1);
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        cout << "Failed to interpereter . . ." << endl;
        exit(-1);
    }
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        cout << "Failed to allocate tensor . . ." << endl;
        exit(-1);
    }
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    int input_type = interpreter->tensor(input)->type;
    input_8 = interpreter->typed_tensor<uint8_t>(input);
    float_32 = interpreter->typed_tensor<float>(input);
}

pair<float,float> TF_Inference::preprocess(Mat& img) {
        cv::Size shape = img.size();
        float scaleRatio = std::min((float)I_HEIGHT / shape.height, (float)I_WIDTH / shape.width);
        
        cv::Size new_unpad = Size(shape.width * scaleRatio, shape.height * scaleRatio);

        auto dw = (I_WIDTH - new_unpad.width) / 2.0;
        auto dh = (I_HEIGHT - new_unpad.height) / 2.0;

        // cout << "dw, dh: " << dw << " " << dh << endl;
        if (shape != new_unpad) {
            cout << "Shape != new_unapd, need to resize" << endl;
            resize(img, img, new_unpad, INTER_LINEAR);
        }
        int top = static_cast<int>(round(dh - 0.1));
        int bottom = static_cast<int>(round(dh + 0.1));
        int left = static_cast<int>(round(dw - 0.1));
        int right = static_cast<int>(round(dw + 0.1));

        copyMakeBorder(img, img, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
        pair<float,float> pad = make_pair(((float)top / shape.height), ((float)left / img.cols));
        // cout << "Image Shape: " << img.size() <<  " " << img.channels() << endl;
        // cout << "Pad Values: " << pad.first << " " << pad.second << endl;


        int matType = img.type();
        int depth = CV_MAT_DEPTH(matType);

        if (depth != CV_32F) {
            img.convertTo(img, CV_32F);
        }

        // int matType2 = img.type();
        // int depth2 = CV_MAT_DEPTH(matType2);

        cvtColor(img, img, cv::COLOR_RGB2BGR); 
        img.convertTo(img, CV_32F, 1.0 / 255.0);
        // YOLO_UTILS::printImageValues(img, 2, 2);
        return pad;
}

void TF_Inference::postProcess(cv::Mat& image, cv::Mat& mat, std::pair<float,float> pad)
{
        cout << pad.first << " " << pad.second << endl;
        mat.row(0) -= pad.second; 
        mat.row(1) -= pad.first; 
        //     // Scale bounding boxes to match the original image size
        float scale_factor = std::max(image.cols, image.rows);
        mat.rowRange(0, 4) *= scale_factor;

        cv::Mat transposed_outputs;
        cv::transpose(mat, transposed_outputs); 
        transposed_outputs.convertTo(transposed_outputs, CV_32F);

        transposed_outputs.col(0) -= transposed_outputs.col(2) / 2; // x_min = center_x - w / 2
        transposed_outputs.col(1) -= transposed_outputs.col(3) / 2; // y_min = center_y - h / 2

        mat = transposed_outputs;

        int num_detections = mat.cols;
        int num_features = mat.rows;

        std::vector<float> scores;
        scores.reserve(mat.rows);
        std::vector<bool> keep;
        std::vector<Rect> boxes;
        std::vector<float>newScores;
        std::vector<int>classIds;
        for (int i = 0; i < mat.rows; i++) {
            float* ptr = mat.ptr<float>(i);
            float maxScore = *std::max_element(ptr + 4, ptr + 84); 
            // scores.push_back(maxScore);
            scores.emplace_back(maxScore);

            if (maxScore > confidence) {
                keep.push_back(maxScore);
                boxes.push_back(Rect(ptr[0], ptr[1], ptr[2], ptr[3]));
                newScores.push_back(maxScore);
                classIds.push_back(std::distance(ptr + 4, std::max_element(ptr + 4, ptr + 84)));
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, newScores, confidence_threshold, nms_threshold, indices); 
        
        std::vector<cv::Scalar> color_palette = YOLO_UTILS::generateColorPalette(80);
        std::string filename = "../classes.txt";
        std::vector<std::string> classes = YOLO_UTILS::loadClasses(filename);
        for (int i : indices) {
            YOLO_UTILS::drawDetections(image, boxes[i], newScores[i], classIds[i], color_palette, classes);
    }


}

void TF_Inference::performInference(cv::Mat &image)
{
    cv::Mat inputImage = image.clone();

    pair<float,float> pad = preprocess(inputImage);
    
    memcpy(float_32, inputImage.ptr<float>(0),
        inputImage.cols * inputImage.rows * 3 * sizeof(float));
    interpreter->Invoke();
    int output = interpreter->outputs()[0];
    // TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    // cout << "OUTPUT DIMS: " << output_dims->size << endl;
    // cout << "Output Size: " << output_dims->data[0] << endl;
    // cout << "Output Size: " << output_dims->data[1] << endl;
    // cout << "Output Size: " << output_dims->data[2] << endl;
    
    float* res = interpreter->typed_output_tensor<float>(0);
    cv::Mat outputMat(84, 8400, CV_32F);
    std::memcpy(outputMat.data, res, 84 * 8400 * sizeof(float));

    postProcess(image, outputMat, pad);
}