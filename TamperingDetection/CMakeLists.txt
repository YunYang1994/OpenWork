cmake_minimum_required(VERSION 3.10)
project(TamperingDetection)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})
add_library(TamperingDetector SHARED "TamperingDetector.cpp")
target_link_libraries(TamperingDetector ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} TamperingDetector)
