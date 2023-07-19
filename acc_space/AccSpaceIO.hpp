#pragma once
#include "npy.hpp"
#include "utils.hpp"
#include "npy_reader.hpp"
#include "options.hpp"
#include "utils.hpp"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
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


typedef e::MatrixXd matrix;
typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;


matrix cvmat_to_eigen(const cv::Mat& mat, const bool& debug = false)
{
    int width = mat.cols;
    int height = mat.rows;
    int channels = mat.channels();

    int type = mat.type();
    int dataType = type & CV_MAT_DEPTH_MASK;

    if (debug) {
        cout << "Converting CVMat to Eigen Matrix" << endl;
        cout << "Data Type: " << type << endl;
        cout << "Channels: " << channels << endl;
    }


    matrix eigenMat;
    if (dataType == CV_8U)
    {
        cout << "Data type: CV_8U" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    #pragma omp critical
                    eigenMat(row, col * channels + c) = static_cast<double>(mat.at<cv::Vec3b>(row, col)[c]);
                }
            }
        }
    }
    else if (dataType == CV_32F)
    {
        cout << "Data type: CV_32F" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    #pragma omp critical
                    eigenMat(row, col * channels + c) = static_cast<double>(mat.at<cv::Vec3f>(row, col)[c]);
                }
            }
        }
    }
    else if (dataType == CV_64F)
    {
        cout << "Data type: CV_64F" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    #pragma omp critical
                    eigenMat(row, col * channels + c) = mat.at<cv::Vec3d>(row, col)[c];
                }
            }
        }
    }
    else
    {
        cout << "Error: Datatype unknown.\n Data type: " << dataType << endl;
        return eigenMat;
    }

    return eigenMat;
}

cv::Mat torch_tensor_to_cv_mat(torch::Tensor tensor) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    cv::Mat mat(rows, cols, CV_32FC1, tensor.data_ptr<float>());
    return mat.clone();
}

matrix torch_tensor_to_eigen(torch::Tensor tensor, const bool& debug = false) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    matrix mat(rows, cols);

    if (debug) {
        cout << "Converting Torch Tensor to Eigen Matrix" << endl;
    }

    auto tensor_accessor = tensor.accessor<double, 2>();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            #pragma omp critical
            mat(i, j) = tensor_accessor[i][j];
        }
    }

    return mat;
}

pc_ptr read_point_cloud(string path, const bool& debug = false) {
    //geometry:: PointCloud pcv;
    pc_ptr pcv(new geometry::PointCloud);
    vector<double> data;
    vector<unsigned long> shape;
    bool fortran_order;

    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

    int rows = static_cast<int>(shape[0]);
    int cols = static_cast<int>(shape[1]);

    if (debug) {
        cout << "Point Cloud shape: " << rows << " " << cols << endl;
    }

    matrix mat(rows, cols);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            #pragma omp critical
            mat(i, j) = data[i * cols + j];
        }
    }

    for (int i = 0; i < mat.rows(); i++) {
        pcv->points_.push_back(e::Vector3d(mat(i, 0), mat(i, 1), mat(i, 2)));
    }

    //for (int i = 0; i < 10; i++) {
    //	cout << "x: " << pcv->points_[i].x() << " y: " << pcv->points_[i].y() << " z: " << pcv->points_[i].z() << endl;
    //}

    return pcv;
}

vector<vector<double>> read_key_points(string path, const bool debug = false) {
    vector<double> data;
    vector<unsigned long> shape;
    bool fortran_order;

    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

    int rows = static_cast<int>(shape[0]);
    int cols = static_cast<int>(shape[1]);

    if (debug) {
        cout << "Keypoint Shape: " << rows << " " << cols << endl;
    }

    vector<vector<double>> mat(rows, vector<double>(cols));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = data[i * cols + j];
        }
    }
    return mat;
}

vector<vector<double>> read_ground_truth(const string& path, const bool& debug = false) {
    ifstream file(path); // Open the file
    string line;
    vector<vector<double>> data;

    if (file.is_open()) {
        while (getline(file, line)) {
            stringstream ss(line);
            string value;
            vector<double> row;

            while (getline(ss, value, ',')) {
                row.push_back(stod(value)); // Convert string to double and add it to the row
            }

            data.push_back(row); // Add the row to the data vector
        }

        file.close(); // Close the file
    }
    else {
        if (debug)
            cout << "Unable to open file: " << path << endl;
        return {}; // Return an empty vector if file opening fails
    }

    return data;
}

Eigen::MatrixXd read_depth_to_matrix(const string& path, const bool& debug = false) {
    Eigen::MatrixXd depth_image;

    if (path.substr(path.length() - 3) == "dpt") {
        if (debug) {
			cout << "Reading in depth file manually " << endl;
		}
		std::ifstream file(path, std::ios::binary);

        if (file) {
			uint32_t h, w;
			file.read(reinterpret_cast<char*>(&h), sizeof(h));
			file.read(reinterpret_cast<char*>(&w), sizeof(w));

			Eigen::MatrixXd data(h, w);
			file.read(reinterpret_cast<char*>(data.data()), h * w * sizeof(double));

			depth_image = data;
		}

		file.close();
	}
    
    depth_image.transpose();
    
    if (debug) {
        cout << "Depth size: " << depth_image.rows() << " x " << depth_image.cols() << endl;
    }

	return depth_image;
}

cv::Mat eigen_matrix_to_cv_mat(Eigen::MatrixXd matrix, const bool& debug = false) {
    cv::Mat mat(matrix.rows(), matrix.cols(), CV_64FC1);

    if (debug) {
        cout << "Converting Eigen to CV Mat" << endl;
    }

	#pragma omp parallel for collapse(2)
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); ++j) {
			mat.at<double>(i, j) = matrix(i, j);
		}
	}

	return mat;

}

cv::Mat read_depth_to_cv(const std::string& path, const bool& debug = false) {
    cv::Mat depth_image;

    if (path.substr(path.length() - 3) == "dpt") {
        if (debug) {
            cout << "Reading in depth file manually " << endl;
        }
        std::ifstream file(path, std::ios::binary);

        if (file) {
            uint32_t h, w;
            file.read(reinterpret_cast<char*>(&h), sizeof(h));
            file.read(reinterpret_cast<char*>(&w), sizeof(w));

            cv::Mat data(h, w, CV_16UC1);
            file.read(reinterpret_cast<char*>(data.data), h * w * sizeof(uint16_t));

            depth_image = data.clone();
        }

        file.close();
    }
    else {
        if (debug) {
            cout << "Using Opencv to read depth file" << endl;
        }
        depth_image = cv::imread(path, cv::IMREAD_UNCHANGED);
    }

    if (debug) {
        cout << "Depth Image size: " << depth_image.size() << endl;
    }

    return depth_image;
}



Eigen::MatrixXd convertToEigenMatrix(const std::vector<Vertex>& vertices){
    int numVertices = vertices.size();
    Eigen::MatrixXd matrix(numVertices, 3);

    #pragma omp parallel for 
    for (int i = 0; i < numVertices; i++)
    {
        matrix(i, 0) = vertices[i].x;
        matrix(i, 1) = vertices[i].y;
        matrix(i, 2) = vertices[i].z;
    }

    return matrix;
}


Eigen::MatrixXd convertToEigenMatrix(const std::array<std::array<double, 3>, 3>& inputArray)
{
    Eigen::MatrixXd matrix(3,3);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            matrix(i, j) = inputArray[i][j];
        }
    }
    return matrix;
}



