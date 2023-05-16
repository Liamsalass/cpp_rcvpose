#ifndef R_MAP_DATASET_HPP
#define R_MAP_DATASET_HPP

#include <torch/data/datasets.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <utility>
#include <filesystem>



//Figure out how to include hdf5
//#include <hdf5.h>
//#include <npy.hpp>


//Do I need?
// #include <glob.h>

struct myExample {
    torch::Tensor img;
    torch::Tensor lbl;
    torch::Tensor sem_lbl;
};

class RMapDataset : public torch::data::datasets::Dataset<RMapDataset, myExample> {
public:
    using TransformFunction = std::tuple<torch::Tensor, torch::Tensor>(const cv::Mat&, const cv::Mat&);

    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name,
        const std::string& kpt_num,
        TransformFunction& transform
    );

    myExample get(size_t index) override;

    //pass c10::optional because the dataset size may be unknown and could also be null
    c10::optional<size_t> size() const override;

private:
    std::string root_;
    std::string set_;
    std::string obj_name_;
    std::string dname_;
    std::string kpt_num_;
    TransformFunction& transform_;

    std::vector<std::string> ids_;

    std::string h5path_;
    std::string imgpath_;
    std::string radialpath_;
    std::string imgsetpath_;

    //h5::File h5f_;
};

#endif  // R_MAP_DATASET_HPP
