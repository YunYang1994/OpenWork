from libc.stdint cimport uint32_t
from libcpp cimport bool

cdef extern from "student.h":
    cdef cppclass Student:
        Student(char* name, int age)

        void set_score(float* score, int n)
        bool delete_score();

        const uint32_t get_age() const
        const float* get_score() const