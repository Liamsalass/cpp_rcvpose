# CMakeLists.txt: CMake project for test, include source and define
# project specific logic here.

cmake_minimum_required(VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(TARGET_NAME "test")

project(${TARGET_NAME})

set(CMAKE_PREFIX_PATH "C:/libtorch_debug/libtorch")

# Find and include OpenCV, boost, open3d,
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

find_package(Torch REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/../rcvpose")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Add source to this project's executable.
add_executable(${TARGET_NAME} "test.cpp" "test.h")

target_link_libraries(${TARGET_NAME} "${CMAKE_SOURCE_DIR}/../rcvpose/out/build/x64-debug/rcvpose.lib" ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Open3D_LIBRARIES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
endif()

# Copy the DLL after building the target
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../rcvpose/out/build/x64-debug/rcvpose.dll"
        $<TARGET_FILE_DIR:${TARGET_NAME}>
)


if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${TARGET_NAME}
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:${TARGET_NAME}>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
