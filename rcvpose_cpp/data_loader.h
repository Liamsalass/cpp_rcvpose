// Used to create train and test loaders
//TODO:
// - implement transform function
// - check into .npy file structure for transform function
// - implement for ycb (check with prof if necessary) 
// - test loaders


#pragma once
#ifndef DATALOADER_H
#define DATALOADER_H


#include <torch/torch.h>
#include <torch/data/datasets.h>
#include <torch/data/example.h>
#include <torch/utils.h>
#include "RMapDataset.h"
#include <opencv2/core.hpp>
#include "options.hpp"
#include <iostream>
#include <string>
#include <vector>





class RData : public RMapDataset {
public:
	RData(
		const std::string& root,
		const std::string& dname,
		const std::string& set,
		const std::string& obj_name,
		const int kpt_num
	);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transform (const cv::Mat& img, const std::vector<double>& target);

};

std::pair<
	std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>>,
	std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>>
>
get_data_loaders(const Options& opts);


#endif // DATALOADER_H