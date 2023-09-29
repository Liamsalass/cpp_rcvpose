#pragma once

#include "utils.hpp"
#include "options.hpp"
#include "FastFor.cu"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Open3D/Geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/Open3D.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <omp.h> 
#include <atomic>
#include <mutex>
#include <iomanip>


Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double>& vec);

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const std::vector<std::vector<double>>& vec);

Eigen::MatrixXd transformKeyPoint(const Eigen::MatrixXd& keypoint, const Eigen::MatrixXd& RTGT, const bool& debug);

void project(const Eigen::MatrixXd& xyz, const Eigen::MatrixXd& K, const Eigen::MatrixXd& RT, Eigen::MatrixXd& xy, Eigen::MatrixXd& actual_xyz);

std::vector<Vertex> rgbd_to_point_cloud(const std::array<std::array<double, 3>, 3>& K, const cv::Mat& depth);

void divideByLargest(cv::Mat& matrix, const bool& debug);

void normalizeMat(cv::Mat& input, const bool& debug);

Eigen::Vector3d Accumulator_3D(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const bool& debug);




