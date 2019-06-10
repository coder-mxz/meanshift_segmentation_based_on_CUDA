cmake_minimum_required(VERSION 3.9)

################################################################
# Project basic configurations
################################################################
project(cuda_meanshift_seg LANGUAGES C CXX CUDA)
find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)

if(CUDA_FOUND)
    message("CUDA found")
    #turn CUDA_SEPARABLE_COMPILATION on.
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_64_BIT_DEVICE_CODE ON)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        message("CUDA set to debug mode")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -std=c++11")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        message("CUDA set to release mode")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
    endif()
endif()

if(OpenCV_FOUND)
    message("OpenCV found")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

################################################################
# Bundled library path
################################################################
include_directories(include)
include_directories(lib/cimg-2.6.4)
include_directories(lib/eigen-3.3.7)
include_directories(lib/spdlog-1.3.1/include)

################################################################
# Submodules
################################################################
add_subdirectory(source/cuda_flooding)
add_subdirectory(source/cuda_ms_filter)
add_subdirectory(source/cuda_union_find)
add_subdirectory(source/union_find)
add_subdirectory(source/test_main_prog)
add_subdirectory(test)

################################################################
# Binary files
################################################################
option(opt_build_main "Build a main program to test functionalities" OFF)
if(opt_build_main)
    add_executable(main source/main.cpp)
    target_link_libraries(main PRIVATE cuda_flooding)
    target_link_libraries(main PRIVATE cuda_ms_fliter)
    target_link_libraries(main PRIVATE cuda_union_find)
    target_link_libraries(main PRIVATE union_find)
endif()