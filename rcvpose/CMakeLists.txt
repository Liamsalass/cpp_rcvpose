# CMakeLists.txt: CMake project for rcvpose, include source and define
# project specific logic here.
cmake_minimum_required(VERSION 3.8)

project(rcvpose)

set(CMAKE_PREFIX_PATH "C:/libtorch_release/libtorch")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source files to this project's target.
set(SOURCE_FILES
    "rcvpose.cpp"
    "rcvpose.h"
    "data_loader.cpp"
    "data_loader.h"
    "RMapDataset.cpp"
    "RMapDataset.h"
    "trainer.h"
    "trainer.cpp"
    "models/bottleneck.h"
    "models/bottleneck.cpp"
    "models/denseFCNResNet152.h"
    "models/denseFCNResNet152.cpp"
    "models/resFCNResNet152.h"
    "models/resFCNResNet152.cpp"
    "JACOBI.h"
    "JACOBI.c"
    "lmshorn.c"
    "lmshorn.h"
    "accumulator_space.h"
    "accumulator_space.cpp"
    "npy.hpp"
    "AccSpaceMath.h" 
    "AccSpaceMath.cpp"
    "AccSpaceIO.h" 
    "AccSpaceIO.cpp"
)

option(BUILD_DLL "Build DLL" OFF)
if(BUILD_DLL)
    list(REMOVE_ITEM SOURCE_FILES "test.cpp")
    add_library(rcvpose SHARED ${SOURCE_FILES})
else()
    list(APPEND SOURCE_FILES "test.cpp")
    add_executable(rcvpose ${SOURCE_FILES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(rcvpose PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(rcvpose PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Open3D_LIBRARIES})

# Set C++ standard
set_property(TARGET rcvpose PROPERTY CXX_STANDARD 20)
set_property(TARGET rcvpose PROPERTY CXX_STANDARD_REQUIRED ON)
