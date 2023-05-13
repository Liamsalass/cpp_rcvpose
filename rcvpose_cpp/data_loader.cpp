#include "data_loader.h"

RData::RData(
	const std::string& root,
	const std::string& dname,
	const std::string& set, 
	const std::string& obj_name,
	const std::string& kpt_num
) : //Fix RMapDataset constructor
	RMapDataset(root, dname, set, obj_name, kpt_num, &RData::transform)
{

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RData::transform(cv::Mat img, cv::Mat lbl)
{
	
}

c10::optional<size_t> RData::size() const
{
	return ids_.size();
}

std::tuple<torch::data::datasets::Dataset<RData>, torch::data::datasets::Dataset<RData>> get_data_loaders(Options opts) {
	std::vector<std::string> modes = { "train", "test" };
	torch::data::datasets::MapDataset<RData> train_data;
	torch::data::datasets::MapDataset<RData> val_data;



}