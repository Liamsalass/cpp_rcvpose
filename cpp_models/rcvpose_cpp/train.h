#pragma once
#ifndef TRAIN_H
#define TRAIN_H

// Main file for training pipeline

#include "models/denseFCNResNet152.h"
#include "models/resFCNResNet152.h"
#include "AccumulatorSpace.h"
#include "utils.h"
#include <torch/torch.h>
#include "data_loader.h"
#include "options.h"

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
};



class Trainer {
public:
    Trainer(
        RData& train_loader,
        RData& val_loader,
        const Options& options
        // SummaryWriter& vis
    );
private:
    Options options_;
    torch::Device device_;
    DenseFCNResNet152 model_;
    torch::optim::Optimizer optim_;
    RData train_loader_;
    RData val_loader_;
    std::vector<torch::optim::LRScheduler> schedulers_;
    // SummaryWriter vis_;
    int epoch_;
    int iter_;
    int iter_val_;
    int max_iter_;
    float best_acc_mean;
    std::string out_;

    void test();
    void train();

};


#endif // TRAIN_H
