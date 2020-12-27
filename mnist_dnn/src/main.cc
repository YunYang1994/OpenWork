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

#include <iostream>

#include "image.h"
#include "tensor.h"

int main() {
    Image image = imread("/Users/yangyun/Desktop/1.png");
    Tensor tensor(image.gray().data, 28, 28);
    tensor[0][0] = 1;
    std::cout << tensor[0][0] << std::endl;

    Tensor a(3,4);
    Tensor b(4,3);

    a.fill(1);
    b.fill(2);

    a[0][2] = 13.1;
    b[1][2] = 21.1;
    a[1][2] = -1.123;

    Tensor c = a.matmul(b);
    std::cout << c[0][0] << " " << c[2][2] << std::endl;
    std::cout << c[1][1] << " " << c[1][2] << std::endl;

    imwrite("test.png", image);
}
