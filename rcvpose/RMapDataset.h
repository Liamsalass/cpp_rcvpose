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
#include <omp.h>
#include <filesystem>
#include "npy.hpp"


class CustomExample : public torch::data::Example<> {
public:
    CustomExample(torch::Tensor data, torch::Tensor rad1, torch::Tensor rad2, torch::Tensor rad3, torch::Tensor sem_target)
        : data_(data), rad1_(rad1), rad2_(rad2), rad3_(rad3), sem_target_(sem_target) {}

    torch::Tensor data() const  { return data_; }
    torch::Tensor rad_1() const  { return rad1_; }
    torch::Tensor rad_2() const  { return rad2_; }
    torch::Tensor rad_3() const  { return rad3_; }
    torch::Tensor sem_target() const { return sem_target_; }

private:
    torch::Tensor data_, rad1_, rad2_, rad3_, sem_target_;
};

class RMapDataset : public torch::data::datasets::Dataset<RMapDataset, CustomExample> {
public:
    RMapDataset(
        const std::string& root,
        const std::string& dname,
        const std::string& set,
        const std::string& obj_name
    );

    

    // Override get() function to return three tensors at index, the lbl, sem_lbl, and img
    CustomExample get(size_t index) override;

    c10::optional<size_t> size() const override final;

    virtual std::vector<torch::Tensor> transform(cv::Mat& img, cv::Mat& kpt1, cv::Mat& kpt2, cv::Mat& kpt3) = 0;

    std::vector<std::string> ids_;
    std::string h5path_;
    std::string imgpath_;
    std::string radialpath1_;
    std::string radialpath2_;
    std::string radialpath3_;
    std::string imgsetpath_;

private:
    const std::string root_;
    const std::string set_;
    const std::string obj_name_;
    const std::string dname_;

    cv::Mat read_npy(const std::string& path);
};


#endif  // R_MAP_DATASET_HPP

