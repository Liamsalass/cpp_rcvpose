#pragma once
#include "AccSpaceIO.h"
#include "AccSpaceMath.h"
#include "lmshorn.h"
#include "npy.hpp"
#include "models/denseFCNResNet152.h"
#include "utils.hpp"
#include "options.hpp"
#include "ransac.h"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Open3D/Geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/Open3D.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h> 
#include <future>
#include <atomic>



void estimate_6d_pose_lm(const Options& opts, DenseFCNResNet152& model);

void estimate_6d_pose(const Options& opts, DenseFCNResNet152& model, cv::Mat& img, cv::Mat& depth, const std::vector<std::vector<double>>& keypoints, const std::vector<Vertex>& orig_point_cloud);