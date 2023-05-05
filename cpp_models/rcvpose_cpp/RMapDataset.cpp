#include "RMapDataset.h"

using namespace cv;

RMapDataset::RMapDataset(
    const std::string& root,
    const std::string& dname,
    const std::string& set,
    const std::string& obj_name,
    const std::string& kpt_num
    //const torch::transforms::transforms_t& transform
)
    : root_(root),
    set_(set),
    //transform_(transform),
    obj_name_(obj_name),
    dname_(dname),
    kpt_num_(kpt_num)
{
    if (dname_ == "lm") {
        imgpath_ = root_ + "/LINEMOD/" + obj_name_ + "/JPEGImages/%s.jpg";
        radialpath_ = root_ + "/LINEMOD/" + obj_name_ + "/Out_pt" + kpt_num_ + "_dm/%s.npy";
        imgsetpath_ = root_ + "/LINEMOD/" + obj_name_ + "/Split/%s.txt";
    }
    else {
        //YCB
        imgsetpath_ = root_ + "/" + obj_name_ + "/Split/%s.txt";
        //_h5path = root_ + "/" + obj_name_ + ".hdf5";
    }
    std::ifstream file(imgsetpath_ % set_);
    std::string img_id;
    while (std::getline(file, img_id)) {
        ids_.emplace_back(img_id);
    }
}

torch::data::Example<> RMapDataset::get(size_t index) {
    std::string img_id = ids_[index];
    torch::Tensor target_torch;
    torch::Tensor img_torch;
    torch::Tensor sem_target_torch;

    if (dname_ == "lm") {
        std::string radial_path = radialpath_ % img_id;
        cv::Mat cv_radial = cv::imread(radial_path, cv::IMREAD_GRAYSCALE);
        cv::Mat cv_img = cv::imread(imgpath_ % img_id, cv::IMREAD_COLOR);
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
        cv::normalize(cv_radial, cv_radial, 0, 1, cv::NORM_MINMAX, CV_32F);

        cv::Mat cv_sem_radial = cv_radial.clone();
        cv::threshold(cv_sem_radial, cv
