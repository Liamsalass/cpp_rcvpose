#include "data_loader.h"

RData::RData(
	const std::string& root = "../datasets/",
	const std::string& dname = "lm",
	const std::string& set = "train",
	const std::string& obj_name = "ape",
	const int kpt_num = 1
) : 
	RMapDataset(root, dname, set, obj_name, kpt_num) { }


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RData::transform(cv::Mat& img, cv::Mat& target) {

    const std::vector<double> mean = { 0.485, 0.456, 0.406 };
    const std::vector<double> std = { 0.229, 0.224, 0.225 };

	img.convertTo(img, CV_64F);
	img /= 255.0;
	target.convertTo(target, CV_64F);

	//Expand dim if req
	if (target.channels() == 1)
		cv::cvtColor(target, target, cv::COLOR_GRAY2BGR);
	target = target.reshape(1, target.rows);

	for (int i = 0; i < 3; i++) {
		img -= mean[i];
		img /= std[i];
	}

	//Remove last row if odd and last col if odd
	if (img.rows % 2 != 0)
		img = img.rowRange(0, img.rows - 1);
	if (img.cols % 2 != 0)
		img = img.colRange(0, img.cols - 1);

	//Convert to tensor
	cv::Mat img_transposed = img.t();
	torch::Tensor img_tensor = torch::from_blob(img_transposed.data, { img_transposed.rows, img_transposed.cols, img_transposed.channels() }, torch::kFloat);
	torch::Tensor lbl_tensor = torch::from_blob(target.data, { target.rows, target.cols, target.channels() }, torch::kFloat);

	// Convert lbl_tensor to binary tensor
	torch::Tensor sem_lbl_tensor = torch::where(lbl_tensor > 0, torch::ones_like(lbl_tensor), -torch::ones_like(lbl_tensor));

	return std::make_tuple(img_tensor, lbl_tensor, sem_lbl_tensor);
}




