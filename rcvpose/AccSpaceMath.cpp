#include "AccSpaceMath.h"

using namespace std;
using namespace open3d;
using namespace Eigen;

typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;



Eigen::MatrixXd vectorToEigenMatrix(const vector<double>& vec) {
    int size = vec.size();
    Eigen::MatrixXd matrix(1, size);


    for (int i = 0; i < size; i++) {
        matrix(0, i) = vec[i];
    }

    return matrix;
}

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const vector<vector<double>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    Eigen::MatrixXd matrix(rows, cols);

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

    actual_xyz = xyz * (RT.block(0, 0, RT.rows(), 3).transpose());
    MatrixXd RTslice = RT.block(0, 3, RT.rows(), 1).transpose();

    for (int i = 0; i < actual_xyz.rows(); i++) {
        actual_xyz.row(i) += RTslice;
    }

    MatrixXd xy_temp = actual_xyz * K.transpose();

    xy = xy_temp.leftCols(2).array() / xy_temp.col(2).array().replicate(1, 2);
}


vector<Vertex> rgbd_to_point_cloud(const array<array<double, 3>, 3>& K, const cv::Mat& depth) {
    cv::Mat depth64F;
    depth.convertTo(depth64F, CV_64F);  // Convert depth image to CV_64F

    vector<cv::Point> nonzeroPoints;
    cv::findNonZero(depth64F, nonzeroPoints);

    vector<double> zs(nonzeroPoints.size());
    vector<double> xs(nonzeroPoints.size());
    vector<double> ys(nonzeroPoints.size());


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        zs[i] = depth64F.at<double>(v, u);
    }


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        xs[i] = ((u - K[0][2]) * zs[i]) / K[0][0];
        ys[i] = ((v - K[1][2]) * zs[i]) / K[1][1];
    }

    vector<Vertex> pointCloud(nonzeroPoints.size());

    for (int i = 0; i < xs.size(); i++) {
        pointCloud[i].x = xs[i];
        pointCloud[i].y = ys[i];
        pointCloud[i].z = zs[i];
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



void fast_for_cpu(const std::vector<Vertex>& xyz_mm, const std::vector<double>& radial_list_mm, int* VoteMap_3D, const int& vote_map_size) {
    const double factor = (std::pow(3, 0.5) / 4.0);
    const int start = 0;

    #pragma omp parallel for
    for (int count = 0; count < xyz_mm.size(); ++count) {
        const Vertex xyz = xyz_mm[count];
        const int radius = round(radial_list_mm[count]);

        for (int i = start; i < vote_map_size; i++) {
            double x_diff = i - xyz.x;

            for (int j = start; j < vote_map_size; j++) {
                double y_diff = j - xyz.y;

                for (int k = start; k < vote_map_size; k++) {
                    double z_diff = k - xyz.z;
                    double distance = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);

                    if (radius - distance < factor && radius - distance > 0) {
                        int index = i * vote_map_size * vote_map_size + j * vote_map_size + k;
                        VoteMap_3D[index] += 1;
                    }
                }
            }
        }
    }
}

void fast_for_cpu2(const std::vector<Vertex>& xyz_mm, const std::vector<double>& radial_list_mm, int* VoteMap_3D, const int& vote_map_size) {
    const double factor = (std::pow(3, 0.5) / 4.0);
    const int start = 0;

    int num_threads = omp_get_max_threads();
    int chunk_size = xyz_mm.size() / num_threads;

    #pragma omp parallel num_threads(num_threads) 
    {
        int tid = omp_get_thread_num();
        int start_index = tid * chunk_size;
        int end_index = (tid == num_threads - 1) ? xyz_mm.size() : start_index + chunk_size;

        for (int count = start_index; count < end_index; ++count) {
            const Vertex xyz = xyz_mm[count];
            const int radius = round(radial_list_mm[count]);

            for (int i = start; i < vote_map_size; i++) {
                double x_diff = i - xyz.x;

                for (int j = start; j < vote_map_size; j++) {
                    double y_diff = j - xyz.y;

                    for (int k = start; k < vote_map_size; k++) {
                        double z_diff = k - xyz.z;
                        double distance = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);

                        if (radius - distance < factor && radius - distance > 0) {
                            int index = i * vote_map_size * vote_map_size + j * vote_map_size + k;
                            VoteMap_3D[index] += 1;
                        }
                    }
                }
            }
        }
    }
}



Vector3d Accumulator_3D(const vector<Vertex>& xyz, const vector<double>& radial_list, const bool& debug = false) {

    double acc_unit = 5;
    // unit 5mm
    vector<Vertex> xyz_mm(xyz.size());



    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();


    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }


    vector<double> radial_list_mm(radial_list.size());

    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = 0;
    double y_mm_min = 0;
    double z_mm_min = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];

    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    double x_mm_max = 0;
    double y_mm_max = 0;
    double z_mm_max = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_max = max(x_mm_max, xyz_mm[i].x);
        y_mm_max = max(y_mm_max, xyz_mm[i].y);
        z_mm_max = max(z_mm_max, xyz_mm[i].z);
    }

    double xyz_mm_max = max(x_mm_max, max(y_mm_max, z_mm_max));

    int length = static_cast<int>(xyz_mm_max);

    int vote_map_dim = length + static_cast<int>(radius_max);

    int total_size = vote_map_dim * vote_map_dim * vote_map_dim;

    int* VoteMap_3D = new int[total_size]();


    //if (use_cuda && !debug) {
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
    //}

    fast_for_cpu(xyz_mm, radial_list_mm, VoteMap_3D, vote_map_dim);

    vector<Eigen::Vector3i> centers;

    int max_vote = 0;

    for (int i = 0; i < vote_map_dim; i++) {
        for (int j = 0; j < vote_map_dim; j++) {
            for (int k = 0; k < vote_map_dim; k++) {
                int index = i * vote_map_dim * vote_map_dim + j * vote_map_dim + k;
                int current_vote = VoteMap_3D[index];
                if (current_vote > max_vote) {
                    centers.clear();
                    max_vote = current_vote;
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
                else if (current_vote == max_vote) {
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
            }
        }
    }

    delete[] VoteMap_3D;

    if (debug) {
        cout << "\tMax vote: " << max_vote << endl;
        cout << "\tCenter: " << centers[0][0] << " " << centers[0][1] << " " << centers[0][2] << endl;
    }


    Eigen::Vector3d center = centers[0].cast<double>();
    if (zero_boundary < 0) {
        center.array() += zero_boundary;
    }

    center[0] = (center[0] + x_mean_mm + 0.5) * acc_unit;
    center[1] = (center[1] + y_mean_mm + 0.5) * acc_unit;
    center[2] = (center[2] + z_mean_mm + 0.5) * acc_unit;

    return center;
}