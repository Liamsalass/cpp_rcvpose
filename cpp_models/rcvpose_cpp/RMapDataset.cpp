#include "RMapDataset.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

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
