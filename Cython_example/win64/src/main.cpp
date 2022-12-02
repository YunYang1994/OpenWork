/*
 * Copyright 2022 YunYang1994 All Rights Reserved. 
 * @Author: YunYang1994
 * @FilePath: main.cpp
 * @Date: 2022-12-02 12:07:29
 */


#include "image.h"
#include <iostream>

int main()
{
    image im = load_image_stb("data/Rainier3.png", 3);
    std::cout << "width: " << im.w << " height: " << im.h << std::endl;
    save_image_stb(im, "data/Rainier3.save.cp", 0);
    image jm = Resize(im.data, im.w, im.h, im.c, im.w*2, im.h*2);
    save_image_stb(jm, "data/Rainier3.resize.cp", 1);
}