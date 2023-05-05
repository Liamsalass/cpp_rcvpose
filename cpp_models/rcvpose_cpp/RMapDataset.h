#ifndef R_MAP_DATASET_HPP
#define R_MAP_DATASET_HPP

#include <torch/torch.h>
#include <torch/utils.h>
#include <string>
#include <vector>

class RMapDataset : public torch::data::datasets::Dataset<RMapDataset> {
public:
	RMapDataset(
		const std::string& root,
		const std::string& dname,
		const std::string& set,
		const std::string& obj_name,
		const std::string& kpt_num,
		const std::

	);
};



#endif  // R_MAP_DATASET_HPP
