# C++ RCVPose

This README.md document provides instructions for linking, compiling, and running the C++ version of RCVPose.

## Libraries

To successfully compile and run RCVPose, you need the following libraries:
- **libtorch** 2.0.1
    - [PyTorch](https://pytorch.org/get-started/locally/)
- **OpenCV** 4.6.0
    - [OpenCV Releases](https://opencv.org/releases/)
- **Open3D**
    - [Download Open3D](http://www.open3d.org/download/)

The DLLs required for the project can also be found in the `copy_into_build` directory and can be used to build the project.

## Linking and Building

There are three ways to link and build the project using the provided `compile_and_run#.sh` files. Be sure to open the file and adjust the parameters and directory paths for the libraries to ensure smooth execution.

Note that in the three `compile_and_run` files, the input parameters can be customized to achieve desired behavior. For further clarification on these parameters, refer to the source code file `options.hpp`.

To run any of these bash scripts, open up a bash terminal and type the command `bash compile_and_run#.sh` where `#` is the number of the bash script you'd like to run. 

### compile_and_run1.sh

This bash script compiles and runs the RCVPose code using g++. It doesn't rely on copying DLL files into the build directory. Ensure that g++ version is 9.2.0 or higher.

### compile_and_run2.sh

Similar to `compile_and_run1.sh`, this script copies all the DLL files from `copy_into_build` into the build directory.

### compile_and_run3.sh

This script uses CMake mingw32 to link and compile the code. Verify that CMake is set up correctly by checking the `rcvpose/CMakeLists.txt` file for accurate path configurations.

## Running RCVPose

### Operations

RCVPose offers three input operations:
- `train`: Trains the RCVPose ResNet152 backend on the specified dataset, saving the model in the `model_dir` parameter location.
- `validate`: Executes the validation cycle on the entire test dataset, providing ADDs and timing performance metrics.
- `estimate`: Conducts 6D pose estimation for the first 100 images in the dataset, displaying the pixel overlay of the predicted object's location.

### Training Parameters

Before executing the code, specify the following input training parameters. Note that these inputs are required for initialization, but not all are necessary for every operation. For example, `initial_lr` pertains only to the training phase and isn't necessary for the `validate` operation.

```cpp
    // Dataset name ("lm" = LINEMOD)
    std::string dname;
    // Root dataset directory
    std::string root_dataset;
    // Resume training from a checkpoint
    bool resume_train = false;
    // Optimizer to use
    std::string optim = "adam";
    // Batch size
    int batch_size = 1;
    // Class name
    std::string class_name = "ape";
    // Initial learning rate
    double initial_lr = 0.0001;
    // Use reduce on plateau
    // If false, will reduce lr every 70 epoch
    bool reduce_on_plateau = false;
    // Patience for reduce on plateau
    int patience = 10;
    // Directory to save model
    std::string model_dir;
    // Run in Demo mode display images
    bool demo_mode = false;
    // Print out debugging information, useful if code is failing and need to find where
    bool verbose = false;
    // Run in Test Occ mode
    bool test_occ = false;
    // Masking threshold used when masking the semantic output
    // Value must be betwee 0 - 1
    // Smaller values decrease speed and increase accuracy wihtin the range of 0.78 to 0.82. 
    // Any larger or smaller reduces accuracy significantly or decreases speed exponentially
    float mask_threshold = 0.8;
```    


## Help

If you need any help running the code, please feel free to reach out to me at: liam.salass@queensu.ca

