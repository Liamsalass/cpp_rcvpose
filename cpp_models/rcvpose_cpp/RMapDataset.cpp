#include "RMapDataset.h"

using namespace cv;

RMapDataset::RMapDataset(
    const std::string& root,
    const std::string& dname,
    const std::string& set,
    const std::string& obj_name,
    const std::string& kpt_num,
    const std::function<torch::Tensor(torch::Tensor)>& transform
) :
    root_(root),
    set_(set),
    obj_name_(obj_name),
    dname_(dname),
    kpt_num_(kpt_num),
    transform_(transform)
{
    //TODfix casting errors
    //detrmine the correct paths based on dataset name
    std::string imgset_path = root + (dname == "lm" ? "/LINEMOD/" + obj_name + "/Split/" : "/" + obj_name + "/Split/") + set + ".txt";
    std::ifstream infile(imgset_path);
    std::string img_id;

    while (infile >> img_id) {
        ids_.push_back(img_id);
        if (dname == "lm") {
            std::string img_path = root + "/LINEMOD/" + obj_name + "/JPEGImages/" + img_id + ".jpg";
            std::string radial_path = root + "/LINEMOD/" + obj_name + "/Out_pt" + kpt_num + "_dm/" + img_id + "*.npy";
            glob_t result;
            glob(radial_path.c_str(), GLOB_TILDE, NULL, &result);
            for (size_t i = 0; i < result.gl_pathc; i++) {
                radialpaths_.push_back(result.gl_pathv[i]);
            }
            imgpaths_.push_back(img_path);
            globfree(&result);
        }
        else {
            std::string h5_path = root + "/" + obj_name + ".hdf5";
        }
        std::cout << "Loaded " << ids_.size() << " images for dataset " << dname_ << " set " << set_ << " object " << obj_name_ << std::endl;
    }
}


//override the get method to return a single example
// TODO: Check implementation
torch::data::Example<> RMapDataset::get(size_t index)
{
   
}
//override the size method to infer the size of the dataset
c10::optional<size_t> RMapDataset::size() const
{
    //check if the ids_ vector is empty
    if (ids_.empty()) {
		return c10::nullopt;
	}
    return ids_.size();    
}
