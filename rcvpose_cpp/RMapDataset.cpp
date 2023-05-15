#include "RMapDataset.h"



RMapDataset::RMapDataset(
	const std::string& root,
	const std::string& dname,
	const std::string& set, 
	const std::string& obj_name,
	const std::string& kpt_num, 
	TransformFunction& transform
) :
	root_(root),
	dname_(dname),
	set_(set),
	obj_name_(obj_name),
	kpt_num_(kpt_num),
	transform_(transform)
{
	if (dname_ == "lm") {
		imgpath_ = root_ + "/LINEMODE/" + obj_name + "/JPEGImages/";
		radialpath_ = root_ + "/LINEMODE/" + obj_name + "/Out_pt/" + kpt_num + "/_dm/";
		imgsetpath_ = root_ + "/LINEMODE/" + obj_name + "/Split/";
	}
	else if (dname == "ycb") {
		h5path_ = root_ + "/YCB/" + obj_name + "/" ;
		imgpath_ = root_ + "/YCB/" + obj_name + "/Split/ ";
	}
	else {
		std::cout << "Dataset name not recognized" << std::endl;
	}

	std::ifstream f(imgsetpath_);
	std::string line;
	while (std::getline(f, line)) {
		ids_.push_back(line);
	}
}

// Override the get() method
torch::data::Example<> RMapDataset::get(size_t index) {
	auto img_id = ids_[index];
	torch::Tensor target;
	cv::Mat img;
	if (dname_ == "lm") {
		target = torch::from_blob(std::vector<float>(radialpath_ + img_id + ".npy").data(), { 1 });
		img = cv::imread(imgpath_ + img_id + ".jpg");
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	}
	else {
		std::cout << "YCB implementation unfinished" << std::endl;
		// YCB
		// Load data from HDF5 file
	}
	if (transform_) {
		auto [img_torch, target_torch, sem_target_torch] = std::apply(transform_, std::make_tuple(img, target));
		return { img_torch, target_torch };
	}
	return { torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte), target };
}

c10::optional<size_t> RMapDataset::size() const {
	return ids_.size();
}