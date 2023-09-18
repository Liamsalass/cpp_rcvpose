//
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <math.h>
//#include <iostream>
//
//
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