# lhwcv  2017-02-18
project(kmeans)

cmake_minimum_required(VERSION 2.8)

aux_source_directory(src SRC_LIST)

include_directories(include)

find_package(OpenCV REQUIRED)

add_executable(run_kmeans ${SRC_LIST})

target_link_libraries(run_kmeans ${OpenCV_LIBS} )