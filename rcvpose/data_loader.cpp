#include "data_loader.h"

RData::RData(
	const std::string& root = "../datasets/",
	const std::string& dname = "lm",
	const std::string& set = "train",
	const std::string& obj_name = "ape"
) :
	RMapDataset(root, dname, set, obj_name) { }


std::vector<torch::Tensor> RData::transform(cv::Mat& img, cv::Mat& gt1, cv::Mat& gt2, cv::Mat& gt3) {
	const std::vector<double> mean = { 0.485, 0.456, 0.406 };
	const std::vector<double> std = { 0.229, 0.224, 0.225 };

	img.convertTo(img, CV_32FC3);
	img /= 255.0;

	gt1.convertTo(gt1, CV_32FC1);
	gt2.convertTo(gt2, CV_32FC1);
	gt3.convertTo(gt3, CV_32FC1);


	for (int i = 0; i < 3; i++) {
		cv::Mat channel(img.size(), CV_32FC1);
		cv::extractChannel(img, channel, i);
		channel = (channel - mean[i]) / std[i];
		cv::insertChannel(channel, img, i);
	}

	if (img.rows % 2 != 0)
		img = img.rowRange(0, img.rows - 1);
	if (img.cols % 2 != 0)
		img = img.colRange(0, img.cols - 1);


	cv::Mat imgTransposed = img.t();
	cv::Mat gt1Transposed = gt1.t();
	cv::Mat gt2Transposed = gt2.t();
	cv::Mat gt3Transposed = gt3.t();


	torch::Tensor imgTensor = torch::from_blob(imgTransposed.data, { imgTransposed.rows, imgTransposed.cols, imgTransposed.channels() }, torch::kFloat32).clone();
	imgTensor = imgTensor.permute({ 2, 0, 1 });
	torch::Tensor gt1Tensor = torch::from_blob(gt1Transposed.data, { gt1Transposed.channels(), gt1Transposed.rows, gt1Transposed.cols }, torch::kFloat32).clone();
	torch::Tensor gt2Tensor = torch::from_blob(gt2Transposed.data, { gt2Transposed.channels(), gt2Transposed.rows, gt2Transposed.cols }, torch::kFloat32).clone();
	torch::Tensor gt3Tensor = torch::from_blob(gt3Transposed.data, { gt3Transposed.channels(), gt3Transposed.rows, gt3Transposed.cols }, torch::kFloat32).clone();
	torch::Tensor semTensor = torch::where(gt1Tensor > 0, torch::ones_like(gt1Tensor), -torch::ones_like(gt1Tensor));

	std::vector<torch::Tensor> tensors = { imgTensor, gt1Tensor, gt2Tensor, gt3Tensor, semTensor };
	return tensors;
}



