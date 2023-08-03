#pragma once

#include <string>
#include <map>
#include <vector>


struct Options {
    // GPU ID to use
    int gpu_id = -1; // -1 = CPU
    // Dataset name ("lm" = LINEMOD, "ycbv", "tless")
    std::string dname;
    // Root dataset directory
    std::string root_dataset;
    // Resume training from a checkpoint
    bool resume_train = false;
    // Optimizer to use
    std::string optim = "adam";
    // Batch size
    int batch_size = 1;
    // Class name
    std::string class_name = "ape";
    // Initial learning rate
    double initial_lr = 0.0001;
    // Use reduce on plateau
    // If false, will reduce lr every 70 epoch
    bool reduce_on_plateau = false;
    // Patience for reduce on plateau
    int patience = 10;
    // Directory to save model
    std::string model_dir;
    // Run in Demo mode display images
    bool demo_mode = false;
    // Print out debugging information 
    bool verbose = false;
    // Run in Test Occ mode
    bool test_occ = false;
    // Configs
    std::map<std::string, std::vector<float>> cfg;
};

