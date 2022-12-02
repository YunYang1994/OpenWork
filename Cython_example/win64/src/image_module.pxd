
cdef extern from "image.h":
    struct image:
        int w
        int h
        int c
        unsigned char* data

    void  save_image_stb(image im, const char* name, int png)
    image load_image_stb(char* filename, int channels)
    image Resize(unsigned char* data, int src_w, int src_h, int src_c, int w, int h)