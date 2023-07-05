# CMakeList.txt : CMake project for acc_space, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("acc_space")

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.
add_executable (acc_space "accumulator_space.cpp" "accumulator_space.h" "options.hpp" "JACOBI.c" "JACOBI.h" "lmshorn.c" "lmshorn.h" "happly.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET acc_space PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES} ${OpenCV_LIBS})

# TODO: Add tests and install targets if needed.
