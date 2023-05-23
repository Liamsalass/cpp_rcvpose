#pragma once

#include <string>
#include <map>
#include <vector>

#ifdef RCVPOSE_EXPORTS
#define RCVPOSE_API __declspec(dllexport)
#else
#define RCVPOSE_API __declspec(dllimport)
#endif

struct RCVPOSE_API Options {
    // GPU ID to use
    int gpu_id;
    // Dataset name ("lm" = LINEMOD, "ycbv", "tless")
    std::string dname;
    // Root dataset directory
    std::string root_dataset;
    // Resume training from a checkpoint
    bool resume_train;
    // Optimizer to use
    std::string optim;
    // Batch size
    int batch_size;
    // Class name
    std::string class_name;
    // Initial learning rate
    double initial_lr;
    // Number of keypoints
    int kpt_num;
    // Directory to save model
    std::string model_dir;
    // Run in Demo mode
    bool demo_mode;
    // Run in Test Occ mode
    bool test_occ;
    // Configs
    std::map<std::string, std::vector<float>> cfg;

};

