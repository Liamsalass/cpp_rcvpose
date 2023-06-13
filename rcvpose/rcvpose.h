// simple_dll.h : Include file for standard system include files
// or project-specific include files.

#pragma once

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
//#include <warning.h>
#include "options.hpp"
#include <exception>
#include "trainer.h"
#include "data_loader.h"
#include <cuda_runtime.h>

class RCVpose {
public:
    RCVpose(Options options);

    // Constructor
    RCVpose(
        int gpu_id,
        std::string dname,
        std::string root_dataset,
        bool resume_train,
        std::string optim,
        int batch_size,
        std::string class_name,
        double initial_lr,
        int kpt_num,
        std::string model_dir,
        bool demo_mode,
        bool test_occ
    );

    // Default constructor
    RCVpose();

    // Default destructor
    ~RCVpose();

    // Prints a summary of the model
    void summary();

    void train();

    // Evaluates the model on the test set
    void validate();

    void compare_models(std::string model1, std::string model2);

    // Saves the model to specified directory
    void saveModel(std::string path);

    // Tests on a single image and saves the output
    void test_img(std::string img_path, std::string output_path);

    //Tests if specified data is loadable
    int test_loaders();

    void demo();

    // Loads a pretrained model
    void loadModel(std::string path);

private:
    Options opts;
    void init();
    bool can_init();
    torch::DeviceType device_type;
    std::string resume;
    //TrainLoader train_loader;
    //TestLoader test_loader;
};


