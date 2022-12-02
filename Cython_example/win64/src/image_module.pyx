from image_module cimport image, save_image_stb, load_image_stb, Resize
from libc.string cimport memcpy
from libc.stdint cimport uint8_t
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
np.import_array()

""" cvarray data format
'b'         signed integer    
'B'         unsigned integer   # uint8_t
'u'         Unicode character 
'h'         signed short    
'H'         unsigned short  
'i'         signed int   
'I'         unsigned int
'l'         signed long    
'L'         unsigned long  
'q'         signed long long    
'Q'         unsigned long long  
'f'         float    
'd'         double    
"""

def load_image(image_path, channels):
    cdef image im = load_image_stb(image_path.encode('utf-8'), channels)
    cdef uint8_t [:, :, :] data = cvarray(shape=(im.h, im.w, im.c), itemsize=sizeof(uint8_t), format="B")
    memcpy(&data[0,0,0], <unsigned char*> im.data, im.h * im.w * im.c * sizeof(uint8_t))
    return np.array(data, dtype=np.uint8)

def save_image(image_path, np.ndarray[uint8_t, ndim=3, mode="c"] data not None, png=0):
    cdef int h = data.shape[0]
    cdef int w = data.shape[1]
    cdef int c = data.shape[2]
    cdef image im
    im.h = h
    im.w = w
    im.c = c
    im.data = <unsigned char*> np.PyArray_DATA(data)
    save_image_stb(im, image_path.encode('utf-8'), png)
    return True

def resize(np.ndarray[uint8_t, ndim=3, mode="c"] data not None, width, height):
    cdef int h = data.shape[0]
    cdef int w = data.shape[1]
    cdef int c = data.shape[2]
    cdef image im = Resize(<unsigned char*> np.PyArray_DATA(data), w, h, c, width, height)
    cdef uint8_t [:, :, :] out = cvarray(shape=(im.h, im.w, im.c), itemsize=sizeof(uint8_t), format="B")
    memcpy(&out[0,0,0], <unsigned char*> im.data, im.h * im.w * im.c * sizeof(uint8_t))
    return np.array(out, dtype=np.uint8)
