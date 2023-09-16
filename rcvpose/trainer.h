#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "options.hpp"
#include "data_loader.h"
#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/serialize.h>
#include "models/denseFCNResNet152.h"
#include <iostream>
#include <omp.h>
#include <iomanip>
#include <chrono>


class Trainer {
public:
	Trainer(const Options& opts, DenseFCNResNet152& model);
	
	void train(Options& opts, DenseFCNResNet152& model);
	void test();
	

private:
	// Have to nest train_epoch function within train function due to instantiation of dataloaders
	// This means resume training isn't currently functioning
	//void train_epoch();
	//void validate();

	torch::Tensor compute_r_loss(torch::Tensor pred, torch::Tensor gt);
	torch::Tensor compute_geo_constraint(torch::Tensor score_rad_1, torch::Tensor score_rad_2, torch::Tensor score_rad_3, torch::Tensor rad_1, torch::Tensor rad_2, torch::Tensor rad_3);
	void printProgressBar(int current, int total, int width);
	torch::optim::Optimizer* optim;
	torch::nn::L1Loss loss_radial;
	torch::nn::SmoothL1Loss loss_geo;
	torch::nn::L1Loss loss_sem;
	torch::DeviceType device_type;

	int epoch;
	int epochs_without_improvement;
	int starting_epoch;
	int iteration;
	int iteration_val;
	int max_iteration;
	int max_epoch;
	double best_acc_mean;
	std::vector<double> current_lr;
	std::string out;
};

#endif // TRAINER_H