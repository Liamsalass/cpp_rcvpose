// Used to create train and test loaders
//TODO:
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
#include <opencv2/imgproc.hpp>
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

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transform (cv::Mat& img, cv::Mat& target);

};

std::pair<
	std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>>,
	std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>>
>
get_data_loaders(const Options& opts);


#endif // DATALOADER_H