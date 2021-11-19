cmake_minimum_required(VERSION 3.10)
project(solver)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE "Release")

add_executable(${PROJECT_NAME} "solve.cc")
include_directories("/usr/local/Cellar/eigen/3.3.7/include/eigen3/")
