#include "RadiusLM.h"



vector<Mat> project(Mat& xyz, Mat& K, Mat& RT)
{
    // Function to project 3D points onto 2D image plane
    // Input:
    // xyz: [N, 3]
    // K: [3, 3]
    // RT : [3, 4]
    // Output:
    // xy: [N, 2]
    // actual_xyz: [N, 3]


    Mat actual_xyz;

    // Apply rigid transformation to xyz
    actual_xyz = xyz * RT(Rect(0, 0, 3, 3)).t() + RT(Rect(3, 0, 1, 3)).t();

    // Project actual_xyz onto 2D image plane using intrinsic matrix K
    Mat xyz_projected = actual_xyz * K.t();
    Mat xy = Mat(xyz_projected.size(), xyz_projected.type());

    // Normalize projected points
    for (int i = 0; i < xyz_projected.rows; i++) {
        xy.at<double>(i, 0) = xyz_projected.at<double>(i, 0) / xyz_projected.at<double>(i, 2);
        xy.at<double>(i, 1) = xyz_projected.at<double>(i, 1) / xyz_projected.at<double>(i, 2);
    }
    vector<Mat> results{ xy, actual_xyz };
    return results;
}

open3d::geometry::PointCloud rdbd_to_point_cloud(Mat& rgb, Mat& depth)
{
    // Function to convert RGB-D image to point cloud
	// Input:
	// rgb: [H, W, 3]
	// depth: [H, W]
	// Output:
	// pcd: [N, 3]

	// Get height and width of image
	int height = rgb.rows;
	int width = rgb.cols;


	open3d::geometry::PointCloud pcd;


	// Iterate through each pixel in image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
        {

			// Get depth value at pixel
			double d = depth.at<double>(i, j);


			// If depth value is 0, skip pixel 
			// This is to avoid adding points with no depth value to the point cloud
            if (d == 0) {
				continue;
			}


			// Get RGB values at pixel
			Vec3b rgb_values = rgb.at<Vec3b>(i, j);

			// Get x, y, z values of pixel
			double x = (j - linemod_K[0][2]) * d / linemod_K[0][0];
			double y = (i - linemod_K[1][2]) * d / linemod_K[1][1];
			double z = d;

			// Add point to point cloud
			pcd.points_.push_back(Eigen::Vector3d(x, y, z));

			// Add color to point cloud
			pcd.colors_.push_back(Eigen::Vector3d(rgb_values[2], rgb_values[1], rgb_values[0]));
		}
	}
	// Return point cloud
	return pcd;
}

