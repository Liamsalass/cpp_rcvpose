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
#include <iomanip>



Eigen::Vector3d Ransac_3D(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug, std::atomic<bool>& flag);


Eigen::Vector3d Ransac_3D_debug(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug);