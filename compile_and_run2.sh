#!/bin/bash
# Used to compile using g++

# Set paths to necessary libraries
opencv_include_dir="C:/opncv/build/include"
libtorch_include_dir="C:/libtorch_release/libtorch/include"
open3d_include_dir="C:/open3d/include"

# Create the build directory if it doesn't exist
mkdir -p rcvpose_build

# Compile the source code
g++ rcvpose/test.cpp -o rcvpose_build/rcvpose -std=c++20 \
    -I"$opencv_include_dir" \
    -I"$libtorch_include_dir" \
    -I"$open3d_include_dir" \
    -I"copy_into_build"

# Copy DLLs to the build directory
cp copy_into_build/*.dll rcvpose_build/

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
./rcvpose_build/rcvpose "$operation" "$dataset_name" "$dataset_root" "$model_directory" "$resume_train" "$optim" "$batch_size" "$class_name" "$initial_lr" "$reduce_on_plateau" "$patience" "$demo_mode" "$verbose" "$test_occ"
