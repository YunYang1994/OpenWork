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

   void fill(float value);                  // 用常数填充矩阵
   Tensor div(float value);                 // 矩阵除以一个常数
   Tensor mul(float value);                 // 矩阵乘以一个常数
   Tensor matmul(const Tensor & rhs);       // 矩阵相乘
   Tensor flatten();                        // 抹平矩阵，返回一个[N, ] 维度的矩阵
   Tensor max();                            // 返回矩阵的最大值
   Tensor clone();                          // 返回矩阵的一个副本

public:
   float *data;                             // 矩阵数据的地址
   int rows, cols;                          // 矩阵的行和列
};

