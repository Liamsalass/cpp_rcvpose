cmake_minimum_required(VERSION 3.8)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(TARGET_NAME "acc_space")

project(${TARGET_NAME})

set(CMAKE_PREFIX_PATH "C:/libtorch_release/libtorch")
set(TORCH_DIR "C:/libtorch_release/libtorch")

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(${TARGET_NAME}
  "accumulator_space.cpp"
  "accumulator_space.h"
  "options.hpp"
  "JACOBI.c"
  "JACOBI.h"
  "lmshorn.c"
  "lmshorn.h"
  "npy.hpp"
  "utils.hpp"
  "models/denseFCNResNet152.h"
  "models/denseFCNResNet152.cpp"
  "models/bottleneck.h"
  "models/bottleneck.cpp"
  "AccSpaceIO.hpp"
  "AccSpaceMath.hpp"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
endif()

set(CMAKE_BUILD_TYPE=Release)

target_link_libraries(${PROJECT_NAME} PRIVATE ${Open3D_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:${TARGET_NAME}>
  )
endif(MSVC)
