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
    if (dname == "lm") {
        //LM
        std::sprintf(imgpath_, "%s/%s/%s/%s/%s/%s/%%s.png", root.c_str(), dname.c_str(), obj_name.c_str(), set.c_str(), kpt_num.c_str(), "%s");
        std::sprintf(radialpath_, "%s/%s/%s/%s/%s/%s/%%s.png", root.c_str(), dname.c_str(), obj_name.c_str(), set.c_str(), kpt_num.c_str(), "%s");
        std::sprintf(imgsetpath_, "%s/%s/%s/%s/%s/%s.txt", root.c_str(), dname.c_str(), obj_name.c_str(), set.c_str(), kpt_num.c_str(), "%s");
    }
    else {
        //YCB
        //std::sprintf(_h5path, "%s/%s.hdf5", root.c_str(), obj_name.c_str());
        std::sprintf(imgsetpath_, "%s/%s/Split/%s.txt", root.c_str(), obj_name.c_str(), "%s");
        //h5f = new h5py.File(_h5path.c_str(), "r");
    }
    std::ifstream infile(std::sprintf(imgsetpath_, set.c_str()));
    std::string line;
    while (std::getline(infile, line)) {
        ids_.push_back(line);
    }


    std::cout << "Loaded " << ids_.size() << " images for dataset " << dname_ << " set " << set_ << " object " << obj_name_ << std::endl;
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
