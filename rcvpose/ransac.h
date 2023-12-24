#pragma once

#include "utils.hpp"
#include "options.hpp"
#include "AccSpaceMath.h"
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


Eigen::Vector3d Ransac_3D(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d Ransac_3D_greenspan(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d Hash_Vote(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d Ransac_Accumulator(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);

Eigen::Vector3d RANSAC_3D_3(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const int& iterations, const bool& debug);
