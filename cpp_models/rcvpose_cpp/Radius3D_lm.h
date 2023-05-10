#pragma once
#ifndef RADIUS3DLM_H
#define RADIUS3DLM_H

#include <iostream>
#include <vector>

std::vector<std::string> linemod_cls_names = { "ape", "benchvise", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp", "phone" };

Eigen::Matrix3f linemod_K = Eigen::Matrix3f::Zero();
bool depthGeneration = false;
std::string linemod_path = "datasets/LINEMOD/";
std::string original_linemod_path = "datasets/LINEMOD_ORIG/";


#endif //RADIUS3DLM_H