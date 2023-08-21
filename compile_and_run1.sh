#!/bin/bash
# Used to compile using g++

# Set paths to necessary libraries
#opencv_include_dir="/path/to/opencv/include"
#opencv_lib_dir="/path/to/opencv/lib"
#libtorch_include_dir="/path/to/libtorch/include"
#libtorch_lib_dir="/path/to/libtorch/lib"
#open3d_include_dir="/path/to/open3d/include"
#open3d_lib_dir="/path/to/open3d/lib"

opencv_include_dir="C:/opncv/build/include"
opencv_lib_dir="C:/opncv/build/x64/vc15/lib"
libtorch_include_dir="C:/libtorch_release/libtorch/include"
libtorch_lib_dir="C:/libtorch_release/libtorch/lib"
open3d_include_dir="C:/open3d/include"
open3d_lib_dir="C:/open3d/lib"

mkdir -p rcvpose_build

# Compile the source code
g++ rcvpose/test.cpp -o rcvpose_build/rcvpose -std=c++20 \
    -I"$opencv_include_dir" -L"$opencv_lib_dir" -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -I"$libtorch_include_dir" -L"$libtorch_lib_dir" -ltorch \
    -I"$open3d_include_dir" -L"$open3d_lib_dir" -lopen3d_core -lopen3d_visualization

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"
else
    echo "Compilation failed"
    exit 1
fi

# Set up testing options parameters
operation="train"

dataset_name="lm"
dataset_root="C:/Users/User/.cw/work/datasets/test"
model_directory="C:/Users/User/.cw/work/cpp_rcvpose/gpu_models/ape"
resume_train="true"
optim="adam"
batch_size="2"
class_name="ape"
initial_lr="0.0001"
reduce_on_plateau="true"
patience="10"
demo_mode="true"
verbose="false"
test_occ="false"

# Run the executable with testing options parameters
./build/rcvpose "$operation" "$dataset_name" "$dataset_root" "$model_directory" "$resume_train" "$optim" "$batch_size" "$class_name" "$initial_lr" "$reduce_on_plateau" "$patience" "$demo_mode" "$verbose" "$test_occ"
