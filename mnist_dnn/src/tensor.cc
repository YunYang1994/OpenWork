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

Tensor::Tensor(int m, int n) {
    rows = m;
    cols = n;
    data = new float[m*n];
}

Tensor::Tensor(float *ptr, int m, int n) {
    int s = m * n;
    data = new float[s];
    for (int i=0; i<s; i++) {
        data[i] = *(ptr+i);
    }
}

Tensor::Tensor (const Tensor & rhs) {
    rows = rhs.rows;
    cols = rhs.cols;
    data = new float[rows*cols];
    memcpy(data, rhs.data, (rows*cols) * sizeof(float));
}

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

Tensor::~Tensor() {
    delete[] data;
    data = nullptr;
}

float* Tensor::operator[](int i) {
    return data + i*cols;
}

void Tensor::fill(float value) {
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        data[i] = value;
    }
}

Tensor Tensor::div(float value) {
    Tensor tensor(rows, cols);
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        tensor.data[i] /= value;
    }
    return tensor;
}

Tensor Tensor::mul(float value) {
    Tensor tensor(rows, cols);
    int s = rows * cols;
    for (int i=0; i<s; i++) {
        tensor.data[i] *= value;
    }
    return tensor;
}

Tensor Tensor::matmul(const Tensor & rhs) {
    assert(cols == rhs.rows);
    Tensor out(rows, rhs.cols);
    out.fill(0.f);

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

Tensor Tensor::clone() {
    Tensor tensor(rows, cols);
    memcpy(tensor.data, data, (rows*cols) * sizeof(float));
    return tensor;
}


