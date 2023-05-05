# CMakeList.txt : CMake project for rcvpose_cpp, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("rcvpose_cpp")


# Add source to this project's executable.
set(CMAKE_PREFIX_PATH ./libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source to this project's executable.
add_executable (rcvpose_cpp "rcvpose_cpp.cpp" "rcvpose_cpp.h" "bottleneck.h" "bottleneck.cpp" "data_loader.h" "data_loader.cpp" "denseFCNResNet152.h" "denseFCNResNet152.cpp" "resFCNResNet152.h" "resFCNResNet152.cpp" "RMapDataset.cpp" "RMapDataset.h")

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET rcvpose_cpp PROPERTY CXX_STANDARD 17)
endif()
