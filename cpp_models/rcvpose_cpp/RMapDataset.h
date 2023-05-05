#ifndef R_MAP_DATASET_HPP
#define R_MAP_DATASET_HPP

#include <torch/torch.h>
#include <torch/utils.h>
#include <string>
#include <vector>


#include <h5pp/h5pp.h>
#include <opencv2/opencv.hpp>

//Figure out how to include h5pp and open cv


class RMapDataset : public torch::data::datasets::Dataset<RMapDataset> {
public:
    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name,
        const std::string& kpt_num
        //onst torch::transforms::transforms_t& transform
    );

    torch::data::Example<> get(size_t index) override;

    c10::optional<size_t> size() const override;

private:
    std::string root_;
    std::string set_;
    //torch::transforms::transforms_t transform_;
    std::string obj_name_;
    std::string dname_;
    std::string kpt_num_;

    std::string imgpath_;
    std::string radialpath_;
    std::string imgsetpath_;
    std::vector<std::string> ids_;
    //h5::File h5f_;
};

#endif  // R_MAP_DATASET_HPP
