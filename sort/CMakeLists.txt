cmake_minimum_required(VERSION 3.10)
project(sort)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS} "./src" "/usr/local/include")

add_executable(${PROJECT_NAME} "./examples/main.cc")
add_library(tracking SHARED "./src/Hungarian.cc" "./src/KalmanTracker.cc" "./src/sort.cc")

target_link_libraries(tracking ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} tracking)
