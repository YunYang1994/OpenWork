from student.student_module cimport Student
from libc.string cimport memcpy
from cython.view cimport array as cvarray
import numpy as np

cdef class Pystudent:
    cdef Student* thisPtr
    cdef int num_score

    def __cinit__(self, name, age):
        self.thisPtr = new Student(name.encode('utf-8'), age)
        self.num_score = 0
    
    def __dealloc__(self):
        del self.thisPtr
    
    def set_score(self, float [:] score, n):
        self.thisPtr.set_score(&score[0], n)
        self.num_score = n

    def get_score(self):
        cdef const float* cp_score = self.thisPtr.get_score()
        cdef float [:] py_score = cvarray(shape=(self.num_score,), itemsize=sizeof(float), format="f")
        memcpy(&py_score[0], <float*>cp_score, self.num_score * sizeof(float))
        return np.array(py_score)
    
    def get_age(self):
        cdef int age = self.thisPtr.get_age()
        return age
    
    
    