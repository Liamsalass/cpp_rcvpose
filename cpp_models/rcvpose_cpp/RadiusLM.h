#pragma once
#ifndef RADIUSLM_H
#define RADIUSLM_H

#include <open3d/geometry/PointCloud.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>
#include <vector>

// For parallelization 
// Look into library documentation for more information
#include <omp.h>

using namespace std;
using namespace cv;

vector<string> linemod_cls_names = { "ape", "benchvise", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp", "phone" };

double linemod_K[3][3] = { {572.4114, 0.0, 325.2611},
                          {0.0, 573.57043, 242.04899},
                          {0.0, 0.0, 1.0} };

bool depthGeneration = false;

string linemod_path = "datasets/LINEMOD/";
string original_linemod_path = "datasets/LINEMOD_ORIG/";

vector<Mat> project(Mat& xyz, Mat& K, Mat& RT);

open3d::geometry::PointCloud rdbd_to_point_cloud(Mat& rgb, Mat& depth);

#endif // RADIUSLM_H


