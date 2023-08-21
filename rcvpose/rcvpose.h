
#pragma once

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include "utils.hpp"
#include <torch/torch.h>
#include "options.hpp"
#include <exception>
#include "trainer.h"
#include <cuda_runtime.h>
#include "models/denseFCNResNet152.h"
#include "accumulator_space.h"
#include "data_loader.h"
#include "AccSpaceIO.h"

class RCVpose {
public:
    //Initializes RCVpose and backend model
    RCVpose(Options& options);

    //Trains model based on training parameters given
    void train();

    // Evaluates the model on the test set
    void validate();

    // Estimates pose for a single image input that has shape (height, width, 4)
    void estimate_pose(cv::Mat img_with_depth_channel);

    // Estimates pose if passed basic C pointers to images
    // Img must have the shape (height, width, 3)
    // Depth must have the shape (height, width, 1)
    void estimate_pose(double* img, double* depth, const int height, const int width);

    // Estimates pose if passed two cv::Mat objects, one img and the other depth
    // Img must have the shape (height, width, 3)
    // Depth must have the shape (height, width, 1)
    void estimate_pose(cv::Mat img, cv::Mat depth);

    // Estimates pose if passed two file paths, one to the image and the other to the depth
    // Img must have the shape (height, width, 3)
    // Depth must have the shape (height, width, 1)
    void estimate_pose(const std::string& img_path, const std::string& depth);

    void save_all_test_tensors();

private:
    Options opts;
    torch::DeviceType device_type;
    std::string resume;
    DenseFCNResNet152 model;
    //TrainLoader train_loader;
    //TestLoader test_loader;

    void inference();
    cv::Mat img;
    cv::Mat depth;
};


