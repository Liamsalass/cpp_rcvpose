#include "RMapDataset.h"


namespace fs = std::filesystem;

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
		std::cout << "YCB Unfinished" << std::endl;
		h5path_ = root_ + "/YCB/" + obj_name + "/" ;
		imgpath_ = root_ + "/YCB/" + obj_name + "/Split/ ";
	}
	else {
		std::cout << "Dataset name not recognized" << std::endl;
	}

	//Gather all img paths 
	for (const auto& entry : fs::directory_iterator(imgpath_)) {
		if (entry.path().extension() == ".jpg") {
			std::string img_id = entry.path().filename().string();
			ids_.push_back(img_id);
		}
	}
}




// Not an overide get() method because requires return of tuple of tensors
myExample RMapDataset::get(size_t index) {
	auto img_id = ids_[index];
	cv::Mat img, target;
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> img_data;

	if (dname_ == "lm") {
		//Needs cnpy to read file
		auto data = std::vector<float>(radialpath_ + img_id + ".npy");

		img = cv::imread(imgpath_ + img_id + ".jpg");
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		
	} else {
		std::cout << "YCB implementation unfinished" << std::endl;
		// YCB
		// Load data from HDF5 file
	}

	if (transform_ != nullptr) {
		img_data = transform_(img, target);
		return myExample{ std::get<0>(img_data), std::get<1>(img_data), std::get<2>(img_data) };
	}
	else {
		//If no transform function is provided, return the image and target as is
		//Convert img and target to tensors
		auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
		img_tensor = img_tensor.permute({ 2, 0, 1 }); //convert to CxHxW
		img_tensor = img_tensor.toType(torch::kFloat);
		img_tensor = img_tensor.div(255); //normalize to [0,1]

		auto target_tensor = torch::from_blob(target.data, { target.rows, target.cols, 3 }, torch::kByte);
		target_tensor = target_tensor.permute({ 2, 0, 1 }); //convert to CxHxW
		target_tensor = target_tensor.toType(torch::kFloat);
		target_tensor = target_tensor.div(255); //normalize to [0,1]

		return myExample{ img_tensor, target_tensor, torch::Tensor() };
	}
}

c10::optional<size_t> RMapDataset::size() const {
	return ids_.size();
}