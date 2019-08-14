SET( CMAKE_CXX_FLAGS "-std=c++14 -O3")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(random_sample_consensus)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (random_sample_consensus random_sample_consensus.cpp)
target_link_libraries (random_sample_consensus ${PCL_LIBRARIES})
add_executable (cloud_viewer cloud_viewer.cpp)
target_link_libraries (cloud_viewer ${PCL_LIBRARIES})

