// Used to load train and test data
// Preprocessing and pass to the model

#pragma once
#ifndef DATALOADER_H
#define DATALOADER_H

#include <torch/torch.h>
#include <torch/data/datasets.h>
#include <torch/data/example.h>
#include "RMapDataset.h"
#include <opencv2/core.hpp>
#include "options.h"
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
		const std::string& kpt_num
	);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transform(cv::Mat img, cv::Mat lbl);


private:
	const std::vector<double> mean = { 0.485, 0.456, 0.406 };
	const std::vector<double> std = { 0.229, 0.224, 0.225 };

};

std::tuple<torch::data::datasets::Dataset<RData>, torch::data::datasets::Dataset<RData>> get_data_loaders(Options opts);

#endif // DATALOADER_H