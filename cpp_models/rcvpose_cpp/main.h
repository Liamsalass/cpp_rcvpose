#pragma once


#include "train.h"

struct Options {
    bool resume_train = false;
    bool demo_mode = false;
    bool test_occ = false;
    int gpu_id = -1;
    int batch_size = 4;
    int kpt_num = 1;
    float initial_lr = 1e-4;
    std::string mode = "train";
    std::string dname = "lm";
    std::string root_dataset = "./datasets/LINEMOD";
    std::string optim = "Adam";
    std::string class_name = "ape";
    std::string model_dir = "ckpts/";
};
