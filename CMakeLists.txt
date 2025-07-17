cmake_minimum_required(VERSION 3.28)

project(Practice)

add_executable(
    Practice
    src/main.cpp
    lib/sum.cpp
    lib/complex.cpp
    lib/antenna_array.cpp
)

add_library(AntennaArray SHARED 
    lib/antenna_array.cpp
    lib/sum.cpp
)

install (
    TARGETS AntennaArray
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
