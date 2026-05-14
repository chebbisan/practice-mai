cmake_minimum_required(VERSION 3.28)

project(Practice)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

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
