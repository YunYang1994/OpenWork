/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : tensor.cc
*   Author      : YunYang1994
*   Created date: 2020-12-27 15:09:44
*   Description :
*
*===============================================================*/

#include <string>
#include <assert.h>
#include "tensor.h"

// 申请内存并初始化每个元素为 0.f
Tensor::Tensor(int m, int n) {
    rows = m;
    cols = n;
    data = new float[m*n];
    fill(0.f);
}

// 利用一块内存初始化
Tensor::Tensor(float *ptr, int m, int n) {
    int s = m * n;
    data = new float[s];
    for (int i=0; i<s; i++) {
        data[i] = *(ptr+i);
    }
}

// 拷贝构造函数
Tensor::Tensor (const Tensor & rhs) {
    rows = rhs.rows;
    cols = rhs.cols;
    data = new float[rows*cols];
    memcpy(data, rhs.data, (rows*cols) * sizeof(float));
}

// 赋值函数，将一个 Tensor 深拷贝给另一个已存在的 Tensor
Tensor& Tensor::operator=(const Tensor &rhs) {
    if(&rhs != this) {
        rows = rhs.rows;
        cols = rhs.cols;

        // 必须释放原有的内存，然后再重新申请一块内存
        delete[] data;
        data = new float[rows*cols];
        memcpy(data, rhs.data, (rows*cols) * sizeof(float));
    }
    return *this;
}

// 析构函数释放内存
Tensor::~Tensor() {
    delete[] data;
    data = nullptr;
}

// 重载下标运算符号返回一个指针地址
float* Tensor::operator[](int i) {
    return data + i*cols;
}

// for 循环每个元素赋值 value
void Tensor::fill(float value) {
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        data[i] = value;
    }
}

// for 循环每个元素除以 value
Tensor Tensor::div(float value) {
    Tensor out(rows, cols);
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        out.data[i] /= value;
    }
    return out;
}

// for 循环每个元素乘以 value
Tensor Tensor::mul(float value) {
    Tensor out(rows, cols);
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        out.data[i] *= value;
    }
    return out;
}

// 两个矩阵相乘
Tensor Tensor::matmul(const Tensor & rhs) {
    assert(cols == rhs.rows);
    Tensor out(rows, rhs.cols);

    for (int i=0; i<out.rows; i++) {
        for (int j=0; j<out.cols; j++) {
            for (int k=0; k<cols; k++) {
                int a = i * cols + k;       // (*this)[i][k]
                int b = k * rhs.cols + j;   // rhs[k][j]
                int c = i * out.cols + j;   // out[i][j]
                out.data[c] += (data[a] * rhs.data[b]);
            }
        }
    }
    return out;
}

// 返回该 Tensor 的副本
Tensor Tensor::clone() {
    Tensor out(rows, cols);
    memcpy(out.data, data, (rows*cols) * sizeof(float));
    return out;
}


