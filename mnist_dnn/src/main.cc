/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : main.cc
*   Author      : YunYang1994
*   Created date: 2020-12-27 01:16:30
*   Description :
*
*===============================================================*/

#include "image.h"

int main() {
    Image image = imread("/Users/yangyun/Desktop/1.png");
    imwrite("test.png", image);
}
