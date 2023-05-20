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


// Returns validation and training dataloaders as pair of two unique pointers
// pointers point to the dataloaders of type: torch::data::StatelessDataLoader<RData> and torch::data::StatefulDataLoader<RData>
//std::pair<std::unique_ptr<torch::data::StatelessDataLoader<RData,torch::data::samplers::RandomSampler>>,std::unique_ptr<torch::data::StatelessDataLoader<RData,torch::data::samplers::SequentialSampler>>> get_data_loaders(const Options& opts) {
//	
//	std::vector<std::string> modes = { "train", "test" };
//	
//    // Instantiate the dataset
//	auto train_dataset = RData(opts.root_dataset, opts.dname, modes[0], opts.class_name, opts.kpt_num);
//	auto val_dataset = RData(opts.root_dataset, opts.dname, modes[1], opts.class_name, opts.kpt_num);
//
//    // Instantiate the dataloaders 
//	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
//		std::move(train_dataset), 
//		torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
//	);
//
//	auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//		std::move(val_dataset),
//		torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
//	);
//
//    // Return the dataloaders as a pair
//    return std::make_pair(train_loader, val_loader);
//	
//}


