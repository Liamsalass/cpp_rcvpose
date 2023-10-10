#pragma once

#include <string>
#include <map>
#include <vector>


struct Options {
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
    // Type of frontend voting system to use
    std::string frontend = "ransac";
    // Patience for reduce on plateau
    int patience = 10;
    // Directory to save model
    std::string model_dir;
    // Run in Demo mode display images
    bool demo_mode = false;
    // Print out debugging information, useful if code is failing and need to find where
    bool verbose = false;
    // Run in Test Occ mode
    bool test_occ = false;
    // Masking threshold used when masking the semantic output
    // Value must be betwee 0 - 1
    // Smaller values decrease speed and increase accuracy wihtin the range of 0.78 to 0.82. 
    // Any larger or smaller reduces accuracy significantly or decreases speed exponentially
    float mask_threshold = 0.8;
    // Epsilon value
    double epsilon = 0.01;
    // Configs
    std::map<std::string, std::vector<float>> cfg;
};

