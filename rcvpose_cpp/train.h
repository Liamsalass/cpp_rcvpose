#pragma once
#ifndef TRAIN_H
#define TRAIN_H

// Main file for training pipeline

#include "models/denseFCNResNet152.h"
#include "models/resFCNResNet152.h"
#include "AccumulatorSpace.h"
#include "utils.hpp"
#include <torch/torch.h>
#include "data_loader.h"
#include "options.hpp"

typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>> TrainLoader;
typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>> TestLoader;

class Trainer {
public:
    Trainer(
        TrainLoader& train_loader,
        TestLoader& val_loader,
        const Options& options
        // SummaryWriter& vis
    );

    void test();
    void train();


private:
    Options options_;
    torch::Device device_;
    DenseFCNResNet152 model_;
    //torch::optim::Optimizer optim_;
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
};


#endif // TRAIN_H
