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

torch::data::Example<> RMapDataset::get(size_t index)
{
    return torch::data::Example<>();
}

c10::optional<size_t> RMapDataset::size() const
{
    return c10::optional<size_t>();
}
