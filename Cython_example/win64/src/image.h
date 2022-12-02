/*
 * Copyright 2022 YunYang1994 All Rights Reserved. 
 * @Author: YunYang1994
 * @FilePath: image.h
 * @Date: 2022-12-02 12:07:44
 */



#ifdef _MSC_VER
#ifdef image_EXPORTS                          // 该宏在 windows 下会自动生成
#define IMAGE_EXPORT __declspec(dllexport)
#else
#define IMAGE_EXPORT __declspec(dllimport)
#endif
#else
#define IMAGE_EXPORT __attribute__((visibility("default")))
#endif


typedef IMAGE_EXPORT struct image{
    int w,h,c;
    unsigned char* data;
} image;

IMAGE_EXPORT void  save_image_stb(image im, const char *name, int png);
IMAGE_EXPORT image load_image_stb(char *filename, int channels);
IMAGE_EXPORT image Resize(unsigned char* data, int src_w, int src_h, int src_c, int dst_w, int dst_h);