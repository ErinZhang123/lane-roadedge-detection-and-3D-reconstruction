
cmake_minimum_required(VERSION 2.8)
project(lane_c)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -w")
SET( CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV 3.3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${Boost_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS})

include_directories(
        ${PROJECT_SOURCE_DIR}/include)
link_directories(${PCL_LIBRARY_DIRS})
add_library(LaneNetDetector SHARED
            src/LaneNetDetector.cpp)
target_link_libraries(LaneNetDetector
                      tensorflow
                      tensorflow_framework
                      ${OpenCV_LIBS}
                      ${Boost_LIBRARIES}
                      glog)
add_definitions(${PCL_DEFINITIONS})
add_executable(test_lanenet   src/test_lanenet.cpp)
target_link_libraries(test_lanenet
                      LaneNetDetector
                      ${OpenCV_LIBS}
                      glog
                      ${PCL_LIBRARIES})


add_executable(test_tensorflow   src/test_tensorflow.cpp)
target_link_libraries(test_tensorflow
                      tensorflow
                      tensorflow_framework
                      glog)