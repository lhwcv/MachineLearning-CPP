# lhwcv  2017-02-18
project(ap)

cmake_minimum_required(VERSION 2.8)

aux_source_directory(src SRC_LIST)

include_directories(include)

find_package(OpenCV REQUIRED)

add_executable(run_ap ${SRC_LIST})

target_link_libraries(run_ap ${OpenCV_LIBS} )