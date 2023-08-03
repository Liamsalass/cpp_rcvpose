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

atomic<int> progress(0);
mutex mtx;


Eigen::MatrixXd vectorToEigenMatrix(const vector<double>& vec) {
    int size = vec.size();
    Eigen::MatrixXd matrix(1, size);  

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        matrix(0, i) = vec[i];  
    }

    return matrix;
}

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const vector<vector<double>>& vec) {
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


void project(const MatrixXd& xyz, const MatrixXd& K, const MatrixXd& RT, MatrixXd& xy, MatrixXd& actual_xyz)
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


open3d::geometry::PointCloud rgbd_to_point_cloud(const array<array<double, 3>, 3>& K, const cv::Mat& depth) {
    cv::Mat depth64F;
    depth.convertTo(depth64F, CV_64F);  // Convert depth image to CV_64F

    vector<cv::Point> nonzeroPoints;
    cv::findNonZero(depth64F, nonzeroPoints);

    vector<double> zs(nonzeroPoints.size());
    vector<double> xs(nonzeroPoints.size());
    vector<double> ys(nonzeroPoints.size());

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
}

//__global__ void cuda_internal(const double* xyz_mm, const double* radial_list_mm, int num_points, double* VoteMap_3D, int map_size_x, int map_size_y, int map_size_z) {
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid < num_points) {
//        double xyz[3] = { xyz_mm[tid * 3], xyz_mm[tid * 3 + 1], xyz_mm[tid * 3 + 2] };
//        double radius = radial_list_mm[tid];
//        double factor = (3.0 * sqrt(3.0)) / 4.0;
//
//        for (int i = 0; i < map_size_x; ++i) {
//            for (int j = 0; j < map_size_y; ++j) {
//                for (int k = 0; k < map_size_z; ++k) {
//                    double distance = sqrt(pow(i - xyz[0], 2) + pow(j - xyz[1], 2) + pow(k - xyz[2], 2));
//                    if (radius - distance < factor && radius - distance >= 0) {
//                        //Race condition error, figure out how to atomic add
//                        VoteMap_3D[i * map_size_y * map_size_z + j * map_size_z + k] += 1;                                         
//                    }
//                }
//            }
//        }
//    }
//}




void fast_for1(const Eigen::MatrixXd& xyz_mm, const Eigen::VectorXd& radial_list_mm, vector<vector<vector<double>>>& VoteMap_3D) {
    double factor = (3.0 * sqrt(3.0)) / 4.0;
    int vote_map_size_i = VoteMap_3D.size();
    int vote_map_size_j = VoteMap_3D[0].size();
    int vote_map_size_k = VoteMap_3D[0][0].size();

    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(static) nowait
        for (int count = 0; count < xyz_mm.rows(); ++count) {

            Eigen::Vector3d xyz = xyz_mm.row(count);
            double radius = round(radial_list_mm(count));
            int i_start = max(0, static_cast<int>(xyz[0] - radius - 1));
            int i_end = min(vote_map_size_i - 1, static_cast<int>(xyz[0] + radius + 1));
            int j_start = max(0, static_cast<int>(xyz[1] - radius - 1));
            int j_end = min(vote_map_size_j - 1, static_cast<int>(xyz[1] + radius + 1));
            int k_start = max(0, static_cast<int>(xyz[2] - radius - 1));
            int k_end = min(vote_map_size_k - 1, static_cast<int>(xyz[2] + radius + 1));

            for (int i = i_start; i <= i_end; ++i) {
                for (int j = j_start; j <= j_end; ++j) {
                    double y_diff = j - xyz[1];
                    double y_diff_sq = y_diff * y_diff;
                    for (int k = k_start; k <= k_end; ++k) {
                        double distance_sq = pow(i - xyz[0], 2) + y_diff_sq + pow(k - xyz[2], 2);
                        double radius_sq = radius * radius;
                        if (radius_sq - distance_sq < factor && radius_sq - distance_sq > 0) {
                            VoteMap_3D[i][j][k] += 1;
                        }
                    }
                }
            }

        }
    }

}

void fast_for2(const Eigen::MatrixXd& xyz_mm, const Eigen::VectorXd& radial_list_mm, vector<vector<vector<double>>>& VoteMap_3D) {
    double factor = (3.0 * sqrt(3.0)) / 4.0;
    int vote_map_size_i = VoteMap_3D.size();
    int vote_map_size_j = VoteMap_3D[0].size();
    int vote_map_size_k = VoteMap_3D[0][0].size();

    // Activate nested parallelism
    omp_set_nested(1);

    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(dynamic) nowait
        for (int count = 0; count < xyz_mm.rows(); ++count) {

            Eigen::Vector3d xyz = xyz_mm.row(count);
            double radius = round(radial_list_mm(count));
            double radius_sq = radius * radius;  

            int i_start = max(0, static_cast<int>(xyz[0] - radius - 1));
            int i_end = min(vote_map_size_i - 1, static_cast<int>(xyz[0] + radius + 1));
            int j_start = max(0, static_cast<int>(xyz[1] - radius - 1));
            int j_end = min(vote_map_size_j - 1, static_cast<int>(xyz[1] + radius + 1));
            int k_start = max(0, static_cast<int>(xyz[2] - radius - 1));
            int k_end = min(vote_map_size_k - 1, static_cast<int>(xyz[2] + radius + 1));

            for (int i = i_start; i <= i_end; ++i) {
                for (int j = j_start; j <= j_end; ++j) {
                    double y_diff = j - xyz[1];
                    double y_diff_sq = y_diff * y_diff;  

                    #pragma omp parallel for  
                    for (int k = k_start; k <= k_end; ++k) {
                        double distance_sq = pow(i - xyz[0], 2) + y_diff_sq + pow(k - xyz[2], 2);
                        if (radius_sq - distance_sq < factor && radius_sq - distance_sq > 0) {
                            VoteMap_3D[i][j][k] += 1;
                        }
                    }
                }
            }

        }
    }

}



Vector3d Accumulator_3D(const geometry::PointCloud& xyz, const vector<double>& radial_list, const bool& use_cuda = true, const bool& debug = false) {

    double acc_unit = 5;
    // unit 5mm
    Eigen::MatrixXd xyz_mm = Eigen::MatrixXd::Zero(xyz.points_.size(), 3);

    #pragma omp parallel for
    for (size_t i = 0; i < xyz.points_.size(); ++i) {
        xyz_mm(i, 0) = xyz.points_[i].x() * 1000 / acc_unit;
        xyz_mm(i, 1) = xyz.points_[i].y() * 1000 / acc_unit;
        xyz_mm(i, 2) = xyz.points_[i].z() * 1000 / acc_unit;
    }
 
    double x_mean_mm = xyz_mm.col(0).mean();
    double y_mean_mm = xyz_mm.col(1).mean();
    double z_mean_mm = xyz_mm.col(2).mean();

    xyz_mm.col(0).array() -= x_mean_mm;
    xyz_mm.col(1).array() -= y_mean_mm;
    xyz_mm.col(2).array() -= z_mean_mm;

    VectorXd radial_list_mm(radial_list.size());
    #pragma omp parallel for
    for (size_t i = 0; i < radial_list.size(); ++i) {
        radial_list_mm(i) = radial_list[i] * 100 / acc_unit;
    }

    double xyz_mm_min = xyz_mm.minCoeff();
    double xyz_mm_max = xyz_mm.maxCoeff();
    double radius_max = radial_list_mm.maxCoeff();

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
        xyz_mm.array() -= zero_boundary;
    }
    int length = static_cast<int>(xyz_mm.maxCoeff());

    if (debug) {
        //Print out size of the votemap3d
        cout << "VoteMap_3D size: " << length + static_cast<int>(radius_max) << endl;
    }

    vector<vector<vector<double>>> VoteMap_3D(length + static_cast<int>(radius_max),vector<vector<double>>(length + static_cast<int>(radius_max),vector<double>(length + static_cast<int>(radius_max),0.0)));

    chrono::steady_clock::time_point tic = chrono::steady_clock::now();
 

    if (use_cuda && !debug) {
        //cout << "Using GPU for fast_for" << endl;
        ////Initialize fast for on GPU
        //double* device_xyz_mm;
        //double* device_radial_list_mm;
        //double* device_VoteMap_3D;
        //
        //cudaMalloc((void**)&device_xyz_mm, xyz_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_radial_list_mm, radial_list_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_VoteMap_3D, VoteMap_3D.size() * sizeof(double));
        //
        //cudaMemcpy(device_xyz_mm, xyz_mm.data(), xyz_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(device_radial_list_mm, radial_list_mm.data(), radial_list_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemset(device_VoteMap_3D, 0, VoteMap_3D.size() * sizeof(double));
        //
        //int num_points = static_cast<int>(xyz.points_.size());
        //
        //int threads_per_block = 256;
        //int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;
        //
        //cuda_internal <<<blocks_per_grid, threads_per_block>>> (device_xyz_mm, device_radial_list_mm, num_points, device_VoteMap_3D, VoteMap_3D.size(), VoteMap_3D[0].size(), VoteMap_3D[0][0].size());
        //
        //cudaMemcpy(VoteMap_3D.data(), device_VoteMap_3D, VoteMap_3D.size() * sizeof(double), cudaMemcpyDeviceToHost);
        //
        //cudaFree(device_xyz_mm);
        //cudaFree(device_radial_list_mm);
        //cudaFree(device_VoteMap_3D);
    }
    else {
        if (debug) {
            cout << "Using CPU for fast_for" << endl;
        }
        fast_for1(xyz_mm, radial_list_mm, VoteMap_3D);
    }    

    chrono::steady_clock::time_point toc = chrono::steady_clock::now();

    if (debug) {
        cout << "Fast For performance: " << chrono::duration_cast<chrono::microseconds>(toc - tic).count() << " microseconds." << endl;
    }


    Eigen::Vector3d center;
    Eigen::Vector3i indices;


    double max_vote = numeric_limits<double>::min();
    for (size_t i = 0; i < VoteMap_3D.size(); ++i) {
        for (size_t j = 0; j < VoteMap_3D[i].size(); ++j) {
            for (size_t k = 0; k < VoteMap_3D[i][j].size(); ++k) {
                if (VoteMap_3D[i][j][k] > max_vote) {
                    max_vote = VoteMap_3D[i][j][k];
                    indices[0] = static_cast<int>(i);
                    indices[1] = static_cast<int>(j);
                    indices[2] = static_cast<int>(k);
                }
            }
        }
    }

    if ((max_vote > 1) && debug) {
        cout << "Multiple centers located." << endl;
    }
    center = indices.cast<double>();
    if (zero_boundary < 0) {
        center.array() += zero_boundary;
    }

    center[0] = (center[0] + x_mean_mm + 0.5) * acc_unit;
    center[1] = (center[1] + y_mean_mm + 0.5) * acc_unit;
    center[2] = (center[2] + z_mean_mm + 0.5) * acc_unit;

    return center;
}