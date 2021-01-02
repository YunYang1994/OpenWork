/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : data.h
*   Author      : YunYang1994
*   Created date: 2021-01-02 21:53:52
*   Description :
*
*===============================================================*/

#pragma once
#include <iostream>
#include "tensor.h"

// 构造一个数据集类
class Data {
public:
    // X 指的是输入神经网络的 image， Y 是该 image 对应的 label
    Tensor X, Y;
    // 构造函数
    Data(Tensor a, Tensor b): X(a), Y(b) { };
    // 由于成员不涉及指针变量，所以拷贝构造函数和析构函数使用默认的就行
    Data(const Data & rhs) = default;
    ~Data() = default;

    // 返回一个 batch_size 为 n 的数据类 Data
    Data randomBatch(int n);
};

Data loadMnistData(char *mnist_path, char *images_file, char *labels_file);
