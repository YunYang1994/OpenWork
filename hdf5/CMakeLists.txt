cmake_minimum_required(VERSION 3.10)
project(demo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}") 

add_executable(${PROJECT_NAME} "main.cpp")
include_directories("${PROJECT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/lib/libhdf5.lib")
target_link_libraries(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/lib/libhdf5_cpp.lib")
