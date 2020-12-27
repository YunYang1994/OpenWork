/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.h
*   Author      : YunYang1994
*   Created date: 2020-12-26 22:12:03
*   Description :
*
*===============================================================*/

#pragma once
#include <string>

struct Image {
    Image(int h, int w, int c);                  // 构造函数
    Image(const Image & rhs);                    // 拷贝构造函数
    ~Image();                                    // 析构函数

    float & at(int y, int x, int z);             // 按照 [H, W, C] 顺序索引像素值

    Image gray();                                // 转灰度图函数
    Image resize(int w, int h);                  // 图像的 resize 操作，最近邻插值

public:
    float *data;
    int rows, cols, channels, size;
};

Image imread(std::string filename);              // 读取图片
void imwrite(std::string filename, Image im);    // 写出图片

