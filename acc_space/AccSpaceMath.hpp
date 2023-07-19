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
using namespace Eigen;

typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;

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


void project(const MatrixXd& xyz, const MatrixXd& K, const MatrixXd& RT, MatrixXd& xy, MatrixXd& actual_xyz, const bool& debug = false)
{
    /*
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    */

    actual_xyz =  xyz * (RT.block(0, 0, RT.rows(), 3).transpose());
    MatrixXd RTslice = RT.block(0, 3, RT.rows(), 1).transpose();

    #pragma omp parallel for
    for (int i = 0; i < actual_xyz.rows(); i++){
        actual_xyz.row(i) += RTslice;
    }

    MatrixXd xy_temp = actual_xyz * K.transpose();

    xy = xy_temp.leftCols(2).array() / xy_temp.col(2).array().replicate(1, 2);
}


open3d::geometry::PointCloud rgbd_to_point_cloud(const std::array<std::array<double, 3>, 3>& K, const cv::Mat& depth) {
    cv::Mat depth64F;
    depth.convertTo(depth64F, CV_64F);  // Convert depth image to CV_64F

    std::vector<cv::Point> nonzeroPoints;
    cv::findNonZero(depth64F, nonzeroPoints);

    std::vector<double> zs(nonzeroPoints.size());
    std::vector<double> xs(nonzeroPoints.size());
    std::vector<double> ys(nonzeroPoints.size());

    #pragma omp parallel for
    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        #pragma omp critical
        {
            v = nonzeroPoints[i].y;
            u = nonzeroPoints[i].x;
        }
        zs[i] = depth64F.at<double>(v, u);
    }

    #pragma omp parallel for
    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        #pragma omp critical
        {
            v = nonzeroPoints[i].y;
            u = nonzeroPoints[i].x;
        }
        xs[i] = ((u - K[0][2]) * zs[i]) / K[0][0];
        ys[i] = ((v - K[1][2]) * zs[i]) / K[1][1];
    }

    open3d::geometry::PointCloud pointCloud;
    pointCloud.points_.resize(xs.size());

    #pragma omp parallel for
    for (int i = 0; i < xs.size(); i++) {
        pointCloud.points_[i] = Eigen::Vector3d(xs[i], ys[i], zs[i]);
    }

    return pointCloud;
}



void divideByLargest(cv::Mat& matrix, const bool& debug = false) {
    double maxVal;
    cv::minMaxLoc(matrix, nullptr, &maxVal);
    if (debug) {
        cout << "Max value in cv::Mat: " << maxVal << endl;
    }
    matrix /= maxVal;
}

void normalizeMat(cv::Mat& input, const bool& debug = false) {
    double minVal, maxVal;
    cv::minMaxLoc(input, &minVal, &maxVal);

    if (debug) {
        cout << "Normalizing data between 0-1" << endl;
        cout << "\tLargest Value: " << maxVal << endl;
        cout << "\tSmallest Value: " << minVal << endl;
    }

    if (maxVal - minVal != 0) {
        double scale = 1.0 / (maxVal - minVal);
        double shift = -minVal / (maxVal - minVal);
        input.convertTo(input, CV_32FC1, scale, shift);
    }
    // If the range is zero, no normalization is performed
}

