cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("main")

set(CMAKE_PREFIX_PATH ./libtorch ;./opencv; ./open3d)

# Find and include OpenCV and h5pp libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

# find_package(h5pp REQUIRED)
# include_directories(${h5pp_INCLUDE_DIRS})

# Add source to this project's executable.


find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source to this project's executable.
add_executable (main
"main.h"
"train.cpp"  
"train.h" 
"train.cpp" 
"models/bottleneck.h" 
"models/bottleneck.cpp" 
"data_loader.h" 
"data_loader.cpp" 
"models/denseFCNResNet152.h"
"models/denseFCNResNet152.cpp" 
"models/resFCNResNet152.h" 
"models/resFCNResNet152.cpp" 
"RMapDataset.cpp" 
"RMapDataset.h" 
"util/horn.cpp" 
"util/horn.h" 
"AccumulatorSpace.h" 
"AccumulatorSpace.cpp"
)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${OpenCV_LIBS} ${Open3D_LIBRARIES})
include_directories(${Open3D_INCLUDE_DIRS})

# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${OpenCV_LIBS} ${h5pp_LIBRARIES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 17)
endif()
