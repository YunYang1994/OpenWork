cmake_minimum_required(VERSION 3.10)
project(demo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}") 

add_executable(${PROJECT_NAME} "main.cpp")

