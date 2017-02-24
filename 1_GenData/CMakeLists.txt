# lhwcv  2017-02-18
project(ml_cv)

aux_source_directory(src SRC_LIST)

include_directories(include)

add_executable(gendata ${SRC_LIST})

target_link_libraries(gendata ${OpenCV_LIBS} )