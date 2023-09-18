#include "RMapDataset.h"


namespace fs = std::filesystem;


RMapDataset::RMapDataset(
	const std::string& root,
	const std::string& dname,
	const std::string& set,
	const std::string& obj_name
) :
	root_(root),
	dname_(dname),
	set_(set),
	obj_name_(obj_name)
{
	if (dname_ == "lm") {
		imgpath_ = root_ + "/LINEMOD/" + obj_name + "/JPEGImages/";
		radialpath1_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt1_dm/";
		radialpath2_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt2_dm/";
		radialpath3_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt3_dm/";
		imgsetpath_ = root_ + "/LINEMOD/" + obj_name + "/Split/%s.txt";
	}
	else if (dname == "ycb") {
		std::cout << "YCB Unfinished" << std::endl;
		h5path_ = root_ + "/YCB/" + obj_name + "/";
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

			if (line.find("synth") != std::string::npos) {
				continue;
			}

			ids_.push_back(line);
		}
		file.close();
	}
	else {
		throw std::runtime_error("Error opening file: " + imgsetpath_);
	}
}


//Overriden get method, if transform is not null, apply transform to img and lbl to return img, lbl, sem_lbl
CustomExample RMapDataset::get(size_t index) {
	std::string img_id = ids_[index];




	cv::Mat img;
	try {
		img = cv::imread(imgpath_ + img_id + ".jpg", cv::IMREAD_COLOR);
	}
	catch (const std::exception& e) {
		std::cout << "Error: Cannot read image\n" << e.what() << std::endl;
	}

	cv::Mat radial_kpt1 = read_npy(radialpath1_ + img_id + ".npy");
	cv::Mat radial_kpt2 = read_npy(radialpath2_ + img_id + ".npy");
	cv::Mat radial_kpt3 = read_npy(radialpath3_ + img_id + ".npy");

	std::vector<torch::Tensor> transfromed_data = transform(img, radial_kpt1, radial_kpt2, radial_kpt3);

	return CustomExample(transfromed_data[0], transfromed_data[1], transfromed_data[2], transfromed_data[3], transfromed_data[4]);
}


c10::optional<size_t> RMapDataset::size() const {
	return ids_.size();
}

cv::Mat RMapDataset::read_npy(const std::string& path)
{
	std::vector<double> data;
	std::vector<unsigned long> shape;
	bool fortran_order;
	try {
		npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
	}
	catch (const std::exception& e) {
		std::cout << "Error: Cannot read " << path << ".npy file\n" << e.what() << std::endl;
	}

	int rows = static_cast<int>(shape[0]);
	int cols = static_cast<int>(shape[1]);
	cv::Mat mat(rows, cols, CV_64F);

	if (fortran_order) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				mat.at<double>(i, j) = data[i + j * rows];
			}
		}
	}
	else {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				mat.at<double>(i, j) = data[i * cols + j];
			}
		}
	}
	return mat;
}


