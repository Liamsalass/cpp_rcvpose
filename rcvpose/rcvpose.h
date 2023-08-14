// simple_dll.h : Include file for standard system include files
// or project-specific include files.

#pragma once

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <filesystem>
#include "utils.hpp"
#include <torch/torch.h>
//#include <warning.h>
#include "options.hpp"
#include <exception>
#include "trainer.h"
#include "models/denseFCNResNet152.h"
#include "data_loader.h"
#include <cuda_runtime.h>

class RCVpose {
public:
    RCVpose(Options& options);

    void train();

    // Evaluates the model on the test set
    void validate();

    void demo();


private:
    Options opts;
    torch::DeviceType device_type;
    std::string resume;
    DenseFCNResNet152 model;
    //TrainLoader train_loader;
    //TestLoader test_loader;
};


