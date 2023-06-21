#pragma once
#include <chrono>
#include <string>
#include "options.hpp"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Open3D/Geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <opencv2/core.hpp>



void estimate_6d_pose_lm(const Options opts);