#ifndef R_MAP_DATASET_HPP
#define R_MAP_DATASET_HPP

#include <torch/torch.h>
#include <torch/utils.h>
#include <string>
#include <vector>


//Figure out how to include h5pp and open cv

#include <h5pp/h5pp.h>
#include <opencv2/opencv.hpp>


class RMapDataset : public torch::data::datasets::Dataset<RMapDataset> {
public:
    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name,
        const std::string& kpt_num,
        const std::function<torch::Tensor(torch::Tensor)>& transform
    );

    torch::data::Example<> get(size_t index) override;

    //pass c10::optional because the dataset size may be unknown and could also be null
    c10::optional<size_t> size() const override;

private:
    std::string root_;
    std::string set_;
    std::string obj_name_;
    std::string dname_;
    std::string kpt_num_;
    std::function<torch::Tensor(torch::Tensor)> transform_;

    std::string imgpath_;
    std::string radialpath_;
    std::string imgsetpath_;
    std::vector<std::string> ids_;
    //h5::File h5f_;
};

#endif  // R_MAP_DATASET_HPP
