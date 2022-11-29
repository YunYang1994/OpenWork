cmake_minimum_required(VERSION 3.13)
set(LIB_NAME student)
set(EXE_NAME demo)

set(STUDENT_SOURCE_FILES
    src/student.h
    src/student.cpp
)

add_library(${LIB_NAME} SHARED ${STUDENT_SOURCE_FILES})

add_executable(${EXE_NAME} src/main.cpp)
target_link_libraries(${EXE_NAME} ${LIB_NAME})

install(TARGETS ${LIB_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/student/lib)
install(FILES src/student.h DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/student/include)