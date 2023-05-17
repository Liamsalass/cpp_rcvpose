#include "data_loader.h"

RData::RData(
	const std::string& root,
	const std::string& dname,
	const std::string& set, 
	const std::string& obj_name,
	const int kpt_num
) : 
	RMapDataset(root, dname, set, obj_name, kpt_num) { }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RData::transform(const cv::Mat& img, const std::vector<double>& target) {

    const std::vector<double> mean = { 0.485, 0.456, 0.406 };
    const std::vector<double> std = { 0.229, 0.224, 0.225 };

    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_64F);
    imgFloat /= 255.0;

    cv::Mat targetFloat(target);
    targetFloat.convertTo(targetFloat, CV_64F);

    if (targetFloat.channels() == 2) {
        cv::Mat tempTarget;
        cv::cvtColor(targetFloat, tempTarget, cv::COLOR_BGR2GRAY);
        cv::threshold(tempTarget, targetFloat, 0, 255, cv::THRESH_BINARY);
    }


    cv::subtract(imgFloat, mean, imgFloat);
    cv::divide(imgFloat, std, imgFloat);

    if (imgFloat.rows % 2 != 0) {
        imgFloat = imgFloat.rowRange(0, imgFloat.rows - 1);
    }

    if (imgFloat.cols % 2 != 0) {
        imgFloat = imgFloat.colRange(0, imgFloat.cols - 1);
    }

    cv::Mat semTarget = targetFloat > 0;
    semTarget.convertTo(semTarget, CV_32F, 1.0, -1.0);

    //To be implemented later
    // if (dname_ != "lm") {
    //     cv::threshold(targetFloat, targetFloat, 10, 0, cv::THRESH_TOZERO_INV);
    // }

    cv::Mat imgTransposed;
    cv::transpose(imgFloat, imgTransposed);

    torch::Tensor imgTensor = torch::from_blob(imgTransposed.data, { imgTransposed.rows, imgTransposed.cols, imgTransposed.channels() }, torch::kFloat).clone();
    torch::Tensor targetTensor = torch::from_blob(targetFloat.data, { targetFloat.rows, targetFloat.cols }, torch::kFloat).clone();
    torch::Tensor semTargetTensor = torch::from_blob(semTarget.data, { semTarget.rows, semTarget.cols }, torch::kFloat).clone();

    return std::make_tuple(imgTensor, targetTensor, semTargetTensor);
}


// Returns validation and training dataloaders as pair of two unique pointers
// pointers point to the dataloaders of type: torch::data::StatelessDataLoader<RData> and torch::data::StatefulDataLoader<RData>
std::pair<
    std::unique_ptr<torch::data::StatelessDataLoader<RData,torch::data::samplers::RandomSampler>>,
    std::unique_ptr<torch::data::StatelessDataLoader<RData,torch::data::samplers::SequentialSampler>>
> get_data_loaders(const Options& opts) {
	
	std::vector<std::string> modes = { "train", "test" };
	
    // Instantiate the dataset
	auto train_dataset = RData(opts.root_dataset, opts.dname, modes[0], opts.class_name, opts.kpt_num);
	auto val_dataset = RData(opts.root_dataset, opts.dname, modes[1], opts.class_name, opts.kpt_num);

    // Instantiate the dataloaders 
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), 
		torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
	);

	auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(val_dataset),
		torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
	);

    // Return the dataloaders as a pair
    return std::make_pair(train_loader, val_loader);
	
}