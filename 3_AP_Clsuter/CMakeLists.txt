# lhwcv  2017-02-18
project(ml_cv)
cmake_minimum_required(VERSION 2.8)


# include directories
INCLUDE_DIRECTORIES(
  include/
)

aux_source_directory(src AP_SRCS)

add_executable(run_ap ${AP_SRCS})

target_link_libraries(run_ap ml_cv_tools_static ${OpenCV_LIBS} )