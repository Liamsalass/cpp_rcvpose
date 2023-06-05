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

	// Convert img and target to floating point format
	img.convertTo(img, CV_32FC3);
	img /= 255.0;
	target.convertTo(target, CV_32FC3);
	target /= 255.0;

	if (target.channels() == 2) {
		// If lbl has two channels, create a new 3-channel BGR image
		cv::Mat expandedLbl;
		cv::cvtColor(target, expandedLbl, cv::COLOR_GRAY2BGR);
		target = expandedLbl.reshape(1, 1); // Reshape expandedLbl to have 1 row and 1 channel
	}

	// Subtract mean and divide by standard deviation
	for (int i = 0; i < 3; i++) {
		cv::Mat channel(img.size(), CV_32FC1);
		cv::extractChannel(img, channel, i);
		channel = (channel - mean[i]) / std[i];
		cv::insertChannel(channel, img, i);
	}

	// Ensure even dimensions
	if (img.rows % 2 != 0)
		img = img.rowRange(0, img.rows - 1);
	if (img.cols % 2 != 0)
		img = img.colRange(0, img.cols - 1);

	// Transpose the matrices
	cv::Mat imgTransposed = img.t();
	cv::Mat targetTransposed = target.t();

	// Create tensors from the transposed matrices
	torch::Tensor imgTensor = torch::from_blob(imgTransposed.data, { imgTransposed.rows, imgTransposed.cols, imgTransposed.channels() }, torch::kFloat32).clone();
	torch::Tensor targetTensor = torch::from_blob(targetTransposed.data, { targetTransposed.rows, targetTransposed.cols, targetTransposed.channels() }, torch::kFloat32).clone();

	// Create semantic label tensor
	torch::Tensor semLblTensor = torch::where(targetTensor > 0, torch::ones_like(targetTensor), -torch::ones_like(targetTensor));

	return std::make_tuple(imgTensor, targetTensor, semLblTensor);
}




cv::Mat RData::get_img(const int idx)
{
	if (ids_.empty()) {
		std::cout << "No Images in ids_" << std::endl;
		return cv::Mat::zeros(1, 1, CV_8UC3); // return black image
	}

	const std::string img = imgpath_ + ids_[idx] + ".jpg";
	std::cout << "Image Path: " << img << std::endl;

	cv::Mat img_mat = cv::imread(img, cv::IMREAD_COLOR);
	return img_mat;
}


cv::Mat RData::get_target(const int idx)
{
	// Check if ids_ is empty
	if (ids_.empty()) {
		std::cout << "No Images in ids_" << std::endl;
		return cv::Mat::zeros(1, 1, CV_8UC3); // Return a black image if ids_ is empty
	}

	std::vector<double> data;
	std::vector<unsigned long> shape;
	bool fortran_order;

	const std::string img = radialpath_ + ids_[idx] + ".npy";
	std::cout << "Radial Path: " << img << std::endl;

	try {
		std::string npy_path = img;
		npy::LoadArrayFromNumpy(npy_path, shape, fortran_order, data);
	}
	catch (const std::runtime_error& error) {
		std::cout << "ERROR: Failed to load radial data" << std::endl;
		std::cout << error.what() << std::endl;
		throw error; // Throw an error if loading the data fails
	}

	// Convert the loaded data to a cv::Mat object
	int rows = static_cast<int>(shape[0]);
	int cols = static_cast<int>(shape[1]);
	cv::Mat target(rows, cols, CV_64F);

	// Assign the data to the cv::Mat object based on the fortran_order
	if (fortran_order) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				target.at<double>(i, j) = data[i + j * rows];
			}
		}
	}
	else {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				target.at<double>(i, j) = data[i * cols + j];
			}
		}
	}

	return target; // Return the converted cv::Mat object
}
