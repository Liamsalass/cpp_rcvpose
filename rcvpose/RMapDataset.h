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
#include <omp.h>



//Figure out how to include hdf5
//#include <hdf5.h>
//#include <npy.hpp>


//Do I need?
// #include <glob.h>

class CustomExample : public torch::data::Example<> {
public:
    CustomExample(torch::Tensor data, torch::Tensor target, torch::Tensor sem_target)
        : data_(data), target_(target), sem_target_(sem_target) {}

    torch::Tensor data() const  { return data_; }
    torch::Tensor target() const  { return target_; }
    torch::Tensor sem_target() const { return sem_target_; }

private:
    torch::Tensor data_, target_, sem_target_;
};

class RMapDataset : public torch::data::datasets::Dataset<RMapDataset, CustomExample> {
public:
    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name,
        const int& kpt_num
    );


    // Override get() function to return three tensors at index, the lbl, sem_lbl, and img
    CustomExample get(size_t index) override;

    c10::optional<size_t> size() const override final;

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transform(cv::Mat& img, cv::Mat& target) = 0;

    std::vector<std::string> ids_;
    std::string h5path_;
    std::string imgpath_;
    std::string radialpath_;
    std::string imgsetpath_;

private:
    const std::string root_;
    const std::string set_;
    const std::string obj_name_;
    const std::string dname_;
    const int kpt_num_;
    //h5::File h5f_;
};


#endif  // R_MAP_DATASET_HPP

