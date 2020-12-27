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
   Tensor(float *ptr, int m, int n);
   Tensor(const Tensor & rhs);              // 拷贝构造函数
   Tensor & operator=(const Tensor & rhs);  // 赋值函数
   ~Tensor();                               // 析构函数

   float* operator[](int i);                // 重载下标运算符，返回一个指针

   void fill(float value);
   Tensor div(float value);
   Tensor mul(float value);
   Tensor matmul(const Tensor & rhs);       // 矩阵相乘
   Tensor flatten();
   Tensor max();
   Tensor clone();

public:
   float *data;
   int rows, cols;
};

