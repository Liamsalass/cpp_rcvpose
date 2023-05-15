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
