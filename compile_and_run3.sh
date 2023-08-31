#!/bin/bash
# Used for compiling using CMAKE

# Set paths to necessary libraries
opencv_include_dir="C:/opencv/build/include"
libtorch_include_dir="C:/libtorch_release/libtorch/include"
open3d_include_dir="C:/open3d/include"

Torch_DIR="C:/libtorch_release/libtorch"

# Create the build directory if it doesn't exist
mkdir -p rcvpose_build

cd rcvpose_build

# Specify the paths to OpenCV, LibTorch, and Open3D libraries using CMake
cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" \
    -DOpenCV_DIR="$opencv_include_dir" \
    -DLibTorch_DIR="$libtorch_include_dir" \
    -DOpen3D_DIR="$open3d_include_dir" \
    ../rcvpose/

cd ..
# Check if DLLs already exist in the build directory
dll_count=$(find rcvpose_build/ -maxdepth 1 -name "*.dll" | wc -l)
dll_count2=$(find copy_into_build/ -maxdepth 1 -name "*.dll" | wc -l)

# Only copy DLLs if they are not equal
if [ "$dll_count" -ne "$dll_count2" ]; then
    cp copy_into_build/*.dll rcvpose_build/
    echo "DLLs copied."
else
    echo "DLLs are already present, skipping copying."
fi

# Move into the build directory
cd rcvpose_build

# Build the project using Make
mingw32-make

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
./rcvpose "$operation" "$dataset_name" "$dataset_root" "$model_directory" "$resume_train" "$optim" "$batch_size" "$class_name" "$initial_lr" "$reduce_on_plateau" "$patience" "$demo_mode" "$verbose" "$test_occ"


