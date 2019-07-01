cmake_minimum_required(VERSION 2.8)
project(lobster)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "/home/beaupreda/Documents/opencv2/share/OpenCV")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(lib_lobster)

set(SOURCE_FILES lib_lobster/BackgroundSubtractorLBSP.cpp
        lib_lobster/BackgroundSubtractorLOBSTER.cpp
        lib_lobster/LBSP.cpp
        lib_lobster/RandUtils.h
        lib_lobster/DistanceUtils.h
        main.cpp)

add_executable(lobster ${SOURCE_FILES})
target_link_libraries(lobster ${OpenCV_LIBS})