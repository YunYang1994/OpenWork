/*
 * Copyright 2022 YunYang1994 All Rights Reserved. 
 * @Author: YunYang1994
 * @FilePath: image.cpp
 * @Date: 2022-12-02 12:08:12
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "image.h"

void free_image(image im)
{
    free(im.data);
}

image make_image(int w, int h, int c)
{
    image out;
    out.h = h;
    out.w = w;
    out.c = c;
    out.data = (unsigned char*)calloc(h*w*c, sizeof(char));
    return out;
}

float get_pixel(image im, int x, int y, int c)
{
    if(x >= im.w) x = im.w - 1;
    if(y >= im.h) y = im.h - 1;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    assert(c >= 0);
    assert(c < im.c);
    float pixel = float(im.data[x + im.w*(y + im.h*c)]);
    return pixel;
}

void set_pixel(image im, int x, int y, int c, float v)
{
    assert(c >= 0);
    assert(c < im.c);
    if(x >= 0 && x < im.w && y >= 0 && y < im.h){
        im.data[x + im.w*(y + im.h*c)] = static_cast<unsigned char>(v);
    }
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int lx = (int) floor(x);
    int ly = (int) floor(y);
    float dx = x - lx;
    float dy = y - ly;
    float v00 = get_pixel(im, lx, ly, c);
    float v10 = get_pixel(im, lx+1, ly, c);
    float v01 = get_pixel(im, lx, ly+1, c);
    float v11 = get_pixel(im, lx+1, ly+1, c);
    float v =   v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + 
                v01*(1-dx)*dy + v11*dx*dy;
    return v;
}

image bilinear_resize(image im, int w, int h)
{
    image r = make_image(w, h, im.c);   
    float xscale = (float)im.w/w;
    float yscale = (float)im.h/h;
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float y = (j+.5)*yscale - .5;
                float x = (i+.5)*xscale - .5;
                float val = bilinear_interpolate(im, x, y, k);
                set_pixel(r, i, j, k, val);
            }
        }
    }
    return r;
}

image Resize(unsigned char* data, int src_w, int src_h, int src_c, int dst_w, int dst_h)
{
    image im = make_image(src_w, src_h, src_c);
    im.data = data;
    image jm = bilinear_resize(im, dst_w, dst_h);
    return jm;
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_image_stb(image im, const char *name, int png)
{
    char buff[256];
    unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = im.data[i + k*im.w*im.h];
        }
    }
    int success = 0;
    if(png){
        sprintf(buff, "%s.png", name);
        success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    } else {
        sprintf(buff, "%s.jpg", name);
        success = stbi_write_jpg(buff, im.w, im.h, im.c, data, 100);
    }
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = data[src_index];
            }
        }
    }
    //We don't like alpha channels, #YOLO
    if(im.c == 4) im.c = 3;
    free(data);
    return im;
}

