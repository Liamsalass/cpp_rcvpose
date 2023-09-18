#pragma once
#include "npy.hpp"
#include "utils.hpp"
#include "options.hpp"
#include "utils.hpp"

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Open3D/Geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/Open3D.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>


Eigen::MatrixXd cvmat_to_eigen(const cv::Mat& mat, const bool& debug);

cv::Mat torch_tensor_to_cv_mat(torch::Tensor tensor);

Eigen::MatrixXd torch_tensor_to_eigen(torch::Tensor tensor, const bool& debug);

std::shared_ptr<open3d::geometry::PointCloud> read_point_cloud(std::string path, const bool& debug);

std::vector<Vertex> read_point_cloud(const std::string& path);

std::vector<std::vector<float>> read_float_npy(std::string path, const bool debug);

std::vector<std::vector<double>> read_double_npy(std::string path, const bool debug);

std::vector<std::vector<double>> read_ground_truth(const std::string& path, const bool& debug);

Eigen::MatrixXd read_depth_to_matrix(const std::string& path, const bool& debug);

cv::Mat eigen_matrix_to_cv_mat(Eigen::MatrixXd matrix, const bool& debug);

torch::Tensor npy_to_tensor(const std::string& path);

cv::Mat read_depth_to_cv(const std::string& path, const bool& debug);

Eigen::MatrixXd convertToEigenMatrix(const std::vector<Vertex>& vertices);

Eigen::MatrixXd convertToEigenMatrix(const std::array<std::array<double, 3>, 3>& inputArray);


