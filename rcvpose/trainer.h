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
#include <torch/serialize.h>
#include "models/denseFCNResNet152.h"
#include <iostream>
#include <iomanip>
#include <chrono>


class Trainer {
public:
	Trainer(Options& options);
	
	void train();
	void test();

	void test_compute_r_loss();
	
	void store_model(std::string path);

	void output_pred(const int& idx, const std::string& path);


private:
	// Have to nest train_epoch function within train function due to instantiation of dataloaders
	// This means resume training isn't currently functioning
	//void train_epoch();
	//void validate();

	torch::Tensor compute_r_loss(torch::Tensor pred, torch::Tensor gt);
	Options& opts;
	DenseFCNResNet152 model;
	torch::optim::Optimizer* optim;
	torch::nn::L1Loss loss_radial;
	torch::nn::L1Loss loss_sem;
	torch::DeviceType device_type;

	void printProgressBar(int current, int total, int width);
	void tensorToFile(const torch::Tensor& tensor, const std::string& filename)

	int epoch;
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