cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


project ("test")
set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-debug-2.0.1+cu118/libtorch")
set(TARGET_NAME test)

# Set the PYTHONHOME environment variable
set(ENV{PYTHONHOME} "C:/Users/User/anaconda3/envs/rcvpose")

# Set the PYTHONPATH environment variable
set(ENV{PYTHONPATH} "C:/Users/User/anaconda3/envs/rcvpose/Lib")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

# find_package(Python REQUIRED)
# include_directories(${Python_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(pybind11 CONFIG REQUIRED)

# Add source to this project's executable.
add_executable (${TARGET_NAME} test.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}" ${OpenCV_LIBS} ${Open3D_LIBRARIES} pybind11::lto pybind11::embed pybind11::module pybind11::headers)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${TARGET_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${TARGET_NAME}>)
endif (MSVC)

set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME_DEBUG ${TARGET_NAME}-debug)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME_RELWITHDEBINFO ${TARGET_NAME}-relwithdebinfo)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME_RELEASE ${TARGET_NAME}-release)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME_MINSIZEREL ${TARGET_NAME}-minsizerel)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")