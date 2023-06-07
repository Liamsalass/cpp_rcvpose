#include "accumulator_space.h"

using namespace std;
using namespace open3d;

vector<string> lm_cls_names = {"ape", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp"};

unordered_map<std::string, double> add_threshold = {
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


std::tuple<Eigen::Matrix3d, Eigen::Matrix4d> project(
    const Eigen::MatrixXd& xyz,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& RT
) 
{
    //Figure out the .block function
    Eigen::MatrixXd actual_xyz = (xyz * RT.block(0, 0, 3, 3).transpose()).colwise() + RT.block(0, 3, 3, 1).transpose();
    Eigen::MatrixXd xy = (actual_xyz * K.transpose()).colwise() / actual_xyz.col(2);
    return std::make_tuple(xy, actual_xyz);
}

Eigen::MatrixXd rgbd_to_point_cloud(
    const Eigen::Matrix3d& K, 
    const Eigen::MatrixXd& depth
) 
{
    std::vector<int> vs, us;
    for (int i = 0; i < depth.rows(); i++) {
        for (int j = 0; j < depth.cols(); j++) {
            if (depth(i, j) != 0) {
                vs.push_back(i);
                us.push_back(j);
            }
        }
    }

    Eigen::MatrixXd pts(vs.size(), 3);
    for (int i = 0; i < vs.size(); i++) {
        double x = ((us[i] - K(0, 2)) * depth(vs[i], us[i])) / K(0, 0);
        double y = ((vs[i] - K(1, 2)) * depth(vs[i], us[i])) / K(1, 1);
        double z = depth(vs[i], us[i]);
        pts.row(i) << x, y, z;
    }

    return pts;
}

Eigen::MatrixXf rgbd_to_color_point_cloud(
    const Eigen::Matrix3d& K,
    const Eigen::MatrixXd& depth,
    const Eigen::MatrixXd& rgb
)
{
    std::vector<int> vs, us;
    for (int i = 0; i < depth.rows(); i++) {
        for (int j = 0; j < depth.cols(); j++) {
            if (depth(i, j) != 0) {
                vs.push_back(i);
                us.push_back(j);
            }
        }
    }

    Eigen::MatrixXf pts(vs.size(), 6);
    for (int i = 0; i < vs.size(); i++) {
        // Static casting to avoid casting errors in Eigen data types (datatype errors)
        float x = static_cast<float>((us[i] - K(0, 2)) * depth(vs[i], us[i]) / K(0, 0));
        float y = static_cast<float>((vs[i] - K(1, 2)) * depth(vs[i], us[i]) / K(1, 1));
        float z = static_cast<float>(depth(vs[i], us[i]));

        //Look at rgb function params
        //float r = static_cast<float>(rgb(vs[i], us[i], 0));
        //float g = static_cast<float>(rgb(vs[i], us[i], 1));
        //float b = static_cast<float>(rgb(vs[i], us[i], 2));
        //pts.row(i) << x, y, z, r, g, b;
    }

    return pts;
}


int main() {
    // Sample usage of the functions

    // Define inputs
    Eigen::MatrixXd xyz(3, 3);  // Example point cloud with 3 points
    xyz << 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0;

    Eigen::Matrix3d K;  // Example intrinsic matrix
    K << 572.4114, 0.0, 325.2611,
        0.0, 573.57043, 242.04899,
        0.0, 0.0, 1.0;

    Eigen::Matrix4d RT;  // Example camera extrinsic matrix
    RT << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

    Eigen::MatrixXd depth(480, 640);  // Example depth image
    // Fill depth image with values

    Eigen::MatrixXd rgb(480, 640, 3);  // Example RGB image


    // Perform point cloud operations
    auto [xy, actual_xyz] = project(xyz, K, RT);
    Eigen::MatrixXd pts = rgbd_to_point_cloud(K, depth);
    Eigen::MatrixXd color_pts = rgbd_to_color_point_cloud(K, depth, rgb);

    // Print results
    std::cout << "Projected XY:\n" << xy << std::endl;
    std::cout << "Actual XYZ:\n" << actual_xyz << std::endl;
    std::cout << "Point cloud from depth:\n" << pts << std::endl;
    std::cout << "Color point cloud from depth and RGB:\n" << color_pts << std::endl;

    return 0;
}



void estimate_6d_pose_lm(const Options opts)
{

}
