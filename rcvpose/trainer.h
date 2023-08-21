#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <filesystem>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/serialize.h>
#include <iostream>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include "options.hpp"
#include "data_loader.h"
#include "utils.hpp"
#include "models/denseFCNResNet152.h"



class Trainer {
public:
	Trainer(const Options& opts, DenseFCNResNet152& model);
	
	void train(Options& opts, DenseFCNResNet152& model);	

private:
	//Functions
	torch::Tensor compute_r_loss(torch::Tensor pred, torch::Tensor gt);
	void printGPUmem();
	cv::Mat tensor_to_mat(torch::Tensor tensor);


	DenseFCNResNet152 model;
	torch::optim::Optimizer* optim;
	torch::nn::L1Loss loss_radial;
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