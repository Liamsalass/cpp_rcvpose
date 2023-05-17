#include "RMapDataset.h"


namespace fs = std::filesystem;


RMapDataset::RMapDataset(
	const std::string& root,
	const std::string& dname,
	const std::string& set, 
	const std::string& obj_name,
	const int& kpt_num
) :
	root_(root),
	dname_(dname),
	set_(set),
	obj_name_(obj_name),
	kpt_num_(kpt_num)
{
	if (dname_ == "lm") {
		imgpath_ = root_ + "/LINEMODE/" + obj_name + "/JPEGImages/";
		radialpath_ = root_ + "/LINEMODE/" + obj_name + "/Out_pt" + std::to_string(kpt_num_) + "_dm/";
		imgsetpath_ = root_ + "/LINEMODE/" + obj_name + "/Split/%s.txt";
	}
	else if (dname == "ycb") {
		std::cout << "YCB Unfinished" << std::endl;
		h5path_ = root_ + "/YCB/" + obj_name + "/" ;
		imgpath_ = root_ + "/YCB/" + obj_name + "/Split/ ";
	}
	else {
		std::cout << "Dataset name not recognized" << std::endl;
	}

	//Gather all img paths if they end with .jpg
	//for (const auto& entry : fs::directory_iterator(imgpath_)) {
	//	if (entry.path().extension() == ".jpg") {
	//		std::string img_id = entry.path().filename().string();
	//		ids_.push_back(img_id);
	//	}
	//}

	std::ifstream file(imgsetpath_.replace(imgsetpath_.find("%s"), 2, set_));
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {

			//remove the \n at the end of the line
			line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

			ids_.push_back(line);
		}
		file.close();
	}
	else {
		throw std::runtime_error("Error opening file: " + imgsetpath_);
	}
}




//Overriden get method, if transform is not null, apply transform to img and lbl to return img, lbl, sem_lbl
myExample RMapDataset::get(size_t index) {
	std::string img_id = ids_[index];

	// TODO:
	//Check type of data and shape stored in radial .npy (may not be float)
	std::vector<double> data;
	std::vector<unsigned long> shape;
	bool fortran_order;

	cv::Mat img, target;

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> img_data;

	if (dname_ == "lm") {
	
		try {
			// Read in image and radial distance map 
			img = cv::imread(imgpath_ + img_id + ".jpg", cv::IMREAD_COLOR);


			// Check implemntation (What is fortran_order?)
			std::string npy_path = radialpath_ + img_id + ".npy";
			npy::LoadArrayFromNumpy(npy_path, shape, fortran_order, data);
		}
		catch (const std::exception& e) {
			std::cout << "Error reading in image: " << imgpath_ + img_id + ".jpg" << std::endl;
			std::cout << "Error reading in radial distance map: " << radialpath_ + img_id + ".npy" << std::endl;
			std::cout << e.what() << std::endl;
		}
	}
	else {
		std::cout << "couldn't find dataset named: " << dname_ << std::endl;
		return myExample{ std::get<0>(img_data), std::get<1>(img_data), std::get<2>(img_data) };
	}

	if (transform) {
		img_data = transform(img, data);
	}
	else {
		std::cout << "No transform function provided" << std::endl;
	}

	return myExample{ std::get<0>(img_data), std::get<1>(img_data), std::get<2>(img_data) };
}

c10::optional<size_t> RMapDataset::size() const {
	return ids_.size();
}

