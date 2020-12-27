/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : tensor.h
*   Author      : YunYang1994
*   Created date: 2020-12-27 13:55:29
*   Description :
*
*===============================================================*/

#pragma once

struct Tensor {
   Tensor(int m, int n);                    // 构造函数
   Tensor(const Tensor & rhs);              // 拷贝构造函数
   Tensor & operator=(const Tensor & rhs);  // 赋值函数
   ~Tensor();                               // 析构函数

   Tensor div(float value);
   Tensor mul(float value);
   Tensor matmul(const Tensor & rhs);
   Tensor flatten();
   Tensor max();
   Tensor clone();

public:
   float *data;
   int rows, cols;
};

