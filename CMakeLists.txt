cmake_minimum_required(VERSION 3.5)
project(sift_match)

FIND_PACKAGE(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        main.cpp env.h)

add_executable(sift_match ${SOURCE_FILES})



target_link_libraries(sift_match ${OpenCV_LIBS} )



