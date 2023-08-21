#!/bin/bash
# Used for compiling using CMAKE

# Compile the source code using CMake
mkdir -p rcvpose_build
cd rcvpose_build

cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../rcvpose/
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


