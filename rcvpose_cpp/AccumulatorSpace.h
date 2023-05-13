#pragma once

#include <iostream>
#include <string>
#include <map>
#include <array>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <open3d/geometry/PointCloud.h>
#include "utils.h"
#include "util/horn.h"
#include "models/denseFCNResNet152.h"
#include "models/resFCNResNet152.h"

typedef std::vector<std::vector<float>> Matrix;

const std::array<std::string, 11> lm_cls_names = { 
    "ape", 
    "cam", 
    "can", 
    "cat", 
    "duck",
    "driller", 
    "eggbox", 
    "glue", 
    "holepuncher",
    "iron",
    "lamp" 
};

const std::array<std::string, 8> lmo_cls_names = {
    "ape", 
    "can", 
    "cat", 
    "duck", 
    "driller",  
    "eggbox", 
    "glue", 
    "holepuncher" 
};

const std::map<int, std::string> ycb_cls_names = {
    {1, "002_master_chef_can"},
    {2, "003_cracker_box"},
    {3, "004_sugar_box"},
    {4, "005_tomato_soup_can"},
    {5, "006_mustard_bottle"},
    {6, "007_tuna_fish_can"},
    {7, "008_pudding_box"},
    {8, "009_gelatin_box"},
    {9, "010_potted_meat_can"},
    {10, "011_banana"},
    {11, "019_pitcher_base"},
    {12, "021_bleach_cleanser"},
    {13, "024_bowl"},
    {14, "025_mug"},
    {15, "035_power_drill"},
    {16, "036_wood_block"},
    {17, "037_scissors"},
    {18, "040_large_marker"},
    {19, "051_large_clamp"},
    {20, "052_extra_large_clamp"},
    {21, "061_foam_brick"} 
};

const std::array<std::string, 2> lm_syms = { "eggbox", "glue" };

const std::array<std::string, 5> ycb_syms = { "024_bowl","036_wood_block","051_large_clamp","052_extra_large_clamp","061_foam_brick" };

const std::map<std::string, float> add_threshold = {
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

const std::array<std::array<double, 3>, 3> linemod_K = { {{572.4114, 0., 325.2611},
                                                  {0., 573.57043, 242.04899},
                                                  {0., 0., 1.}} };


//To be optimized (Should I use an api?)
Matrix dot (Matrix A, Matrix B);

std::vector<Matrix, Matrix> project (Matrix xyz, Matrix, Matrix RT);

open3d::geometry::PointCloud rgbd_to_point_cloud(Matrix K, Matrix depth);

Matrix rgbd_to_color_point_cloud(Matrix K, cv::Mat depth, cv::Mat rgb);

Matrix rgbd_to_point_cloud_no_depth(Matrix K, cv::Mat depth);

Matrix rgbd_to_color_point_cloud_no_depth(Matrix K, cv::Mat depth, cv::Mat rgb);

