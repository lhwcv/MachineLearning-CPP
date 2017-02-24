# lhwcv  2017-02-18
project(ml_cv)
cmake_minimum_required(VERSION 2.8)


# include directories
INCLUDE_DIRECTORIES(
  include/
)

SET(TOOLS_SRCS src/LoadData.cpp)


ADD_LIBRARY(ml_cv_tools SHARED ${TOOLS_SRCS})
ADD_LIBRARY(ml_cv_tools_static STATIC ${TOOLS_SRCS})
target_link_libraries(ml_cv_tools ${OpenCV_LIBS} )
target_link_libraries(ml_cv_tools_static ${OpenCV_LIBS} )