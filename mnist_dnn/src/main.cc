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
    Image image = imread("test.png");
    imwrite("test_2.png", image);
}
