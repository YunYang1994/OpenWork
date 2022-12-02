cmake_minimum_required(VERSION 3.13)
set(LIB_NAME image)
set(EXE_NAME demo)

set(STUDENT_SOURCE_FILES
    src/image.cpp
)

add_library(${LIB_NAME} SHARED ${STUDENT_SOURCE_FILES})

add_executable(${EXE_NAME} src/main.cpp)
target_link_libraries(${EXE_NAME} ${LIB_NAME})

SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)