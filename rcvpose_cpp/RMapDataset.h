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
#include "npy.hpp"
#include <utility>
#include <filesystem>



//Figure out how to include hdf5
//#include <hdf5.h>
//#include <npy.hpp>


//Do I need?
// #include <glob.h>

// get return structure
struct myExample {
    torch::Tensor img;
    torch::Tensor lbl;
    torch::Tensor sem_lbl;
};



class RMapDataset : public torch::data::datasets::Dataset<RMapDataset, myExample> {
public:
    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name,
        const int& kpt_num
    );


    // Override get() function to return three tensors at index, the lbl, sem_lbl, and img
    myExample get(size_t index) override;

    //pass c10::optional because the dataset size may be unknown and could also be null
    c10::optional<size_t> size() const override;

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transform(const cv::Mat& img, const std::vector<double>& target) = 0;

private:
    const std::string root_;
    const std::string set_;
    const std::string obj_name_;
    const std::string dname_;
    const int kpt_num_;

    std::vector<std::string> ids_;

    std::string h5path_;
    std::string imgpath_;
    std::string radialpath_;
    std::string imgsetpath_;

    //h5::File h5f_;
};

#endif  // R_MAP_DATASET_HPP
