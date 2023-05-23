#pragma once
#include <string>
#include <map>
struct Options {
    std::string mode = "train";
    int gpu_id = -1;
    std::string dname = "lm";
    std::string root_dataset = "./datasets/LINEMOD";
    bool resume_train = false;
    std::string optim = "Adam";
    int batch_size = 4;
    std::string class_name = "ape";
    float initial_lr = 1e-4;
    int kpt_num = 1;
    std::string model_dir = "ckpts/";
    bool demo_mode = false;
    bool test_occ = false;
    std::map < std::string, std::vector<float>> cfg;
};