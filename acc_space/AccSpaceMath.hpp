#pragma once
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

using namespace std;
using namespace open3d;
namespace e = Eigen;

typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;
typedef e::MatrixXd matrix;

//vector<string> lm_cls_names = {"ape", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp"};
vector<string> lm_cls_names = { "ape" };

unordered_map<string, double> add_threshold = {
    {"eggbox", 0.019735770122546523},
    {"ape", 0.01421240983190395},
    {"cat", 0.018594838977253875},
    {"cam", 0.02222763033276377},
    {"duck", 0.015569664208967385},
    {"glue", 0.01930723067998101},
    {"can", 0.028415044264086586},
    {"driller", 0.031877906042},
    {"holepuncher", 0.019606109985}
};

array<array<double, 3>, 3> linemod_K {{
    {572.4114, 0.0, 325.2611},
    { 0.0, 573.57043, 242.04899 },
    { 0.0, 0.0, 1.0 }
    }};

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double>& vec) {
    int size = vec.size();
    Eigen::MatrixXd matrix(1, size);  

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        matrix(0, i) = vec[i];  
    }

    return matrix;
}

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const std::vector<std::vector<double>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    Eigen::MatrixXd matrix(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = vec[i][j];
        }
    }

    return matrix;
}

Eigen::MatrixXd transformKeyPoint(const Eigen::MatrixXd& keypoint, const Eigen::MatrixXd& RTGT, const bool& debug) {

    int rows = keypoint.rows();
    int cols = RTGT.cols() - 1;

    if (debug) {
        cout << "Keypoint shape: " << keypoint.rows() << " " << keypoint.cols() << endl;
        cout << "RTGT shape: " << RTGT.rows() << " " << RTGT.cols() << endl;
        cout << "cols: " << cols << endl;
        cout << "rows: " << rows << endl;
    }

    Eigen::MatrixXd keypointTransformed = keypoint * RTGT.block(0, 0, cols, 3).transpose();
    keypointTransformed += RTGT.block(0, 3, cols, 1).transpose();

    return keypointTransformed * 1000.0;
}


void project(const matrix& xyz, const matrix& K, const matrix& RT, matrix& xy, matrix& actual_xyz) {
    // xyz: [N, 3]
    // K: [3, 3]
    // RT: [3, 4]
    matrix xyz_rt = (xyz * RT.block(0, 0, 3, 3).transpose()) + RT.block(0, 3, 3, 1).transpose().replicate(4, 1);
    actual_xyz = xyz_rt;
    matrix xyz_k = xyz_rt * K.transpose();
    xy = xyz_k.leftCols(2).array() / xyz_k.col(2).array().replicate(1, 2);
}


pc_ptr rgbd_to_point_cloud(const matrix& K, const matrix& depth) {
    pc point_cloud;

    e::Index heigh = depth.rows();
    e::Index width = depth.cols();

    pc_ptr cloud(new geometry::PointCloud);

    for (e::Index i = 0; i < heigh; i++) {
        for (e::Index j = 0; j < width; j++) {
            double z = depth(i, j);
            if (z > 0) {
                double x = (j - K(0, 2)) * z / K(0, 0);
                double y = (i - K(1, 2)) * z / K(1, 1);
                point_cloud.points_.push_back(e::Vector3d(x, y, z));
            }
        }
    }

    cloud->points_ = move(point_cloud.points_);
    return cloud;
}