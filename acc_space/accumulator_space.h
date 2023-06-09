#pragma once
#include "AccSpaceIO.hpp"
#include "AccSpaceMath.hpp"
#include "lmshorn.h"
#include "npy.hpp"
#include "happly.h"
#include "models/denseFCNResNet152.h"
#include "utils.hpp"
#include "npy_reader.hpp"
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
#include <Open3D/Geometry/PointCloud.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/Open3D.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h> 



void estimate_6d_pose_lm(const Options opts);