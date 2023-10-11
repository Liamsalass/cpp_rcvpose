#pragma once

#include "utils.hpp"
#include "options.hpp"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <open3d/utility/FileSystem.h>
#include <omp.h> 
#include <atomic>
#include <mutex>
#include <future>
#include <iomanip>

//==========================================================================
//Estimating 6D Pose for LINEMOD
//Estimating for ape
//Number of test images : 1050
//Masking Threshold : 0.8
//Epsilon : 0.01
//ADD : 61.5238 % | 1050 / 1050[===========================================] 100 %
//ADD of ape before ICP : 61.5238
//ADD of ape after ICP : 100
//RANSAC count : 3150
//Acc Space count : 0
//Total Time : 0 hours, 14 minutes, and 39 seconds.
//Avg Time per image : 0.837143 seconds.
//Avg Accumulator time : 0 seconds.
//Avg RANSAC time : 0.0181543 seconds.
//Avg Backend Time : 0.116427 seconds.
//Avg ICP Time : 0.634781 seconds.
//Avg Input Size to Accumulator Space : 3779.7

Eigen::Vector3d Ransac_3D(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d Ransac_3D_greenspan(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d Hash_Vote(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

