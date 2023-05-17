cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set (TARGET_NAME "rcvpose")
project (${TARGET_NAME})

set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-debug-2.0.1+cu118/libtorch")
set(BOOST_ROOT "C:/boost_1_82_0")

# Find and include OpenCV, boost, open3d,
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package (Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})


find_package(Torch REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source to this project's executable.
add_executable (
	${TARGET_NAME}
	"main.cpp"
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
	"RadiusLM.h" 
	"RadiusLM.h" 
	"RadiusLM.cpp"
	"options.h"
	"npy.hpp"
)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${OpenCV_LIBS} ${Open3D_LIBRARIES} ${Boost_LIBS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
endif()
