/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : data.cc
*   Author      : YunYang1994
*   Created date: 2021-01-02 22:20:34
*   Description :
*
*===============================================================*/

#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include "data.h"
#include "image.h"

// 返回一个 batch_size 为 n 的数据类 Data
Data Data::randomBatch(int n) {
    Tensor A(n, X.cols);
    Tensor B(n, Y.cols);
    // 随机选择 n 个样本，将它们的 X 和 Y 分别拷贝至 A 和 B 中
    for (int i = 0; i < n; i++) {
        int j = rand() % X.rows;
        for (int k = 0; k < X.cols; k++) {
            A[i][k] = X[j][k];
        }
        for (int k = 0; k < Y.cols; k++) {
            B[i][k] = Y[j][k];
        }
    }
    return Data(A, B);
}

// 读取 mnist 数据集
Data loadMnistData(const char *mnist_path, const char *images_file) {
    std::fstream fin(images_file, std::ios::in);
    std::vector<std::string> image_paths;

    if (fin.is_open()) {
        std::string line;
        while (std::getline(fin, line)) {
            // 一行一行地解析图片路径
            std::string image_path = mnist_path + line;
            image_paths.push_back(image_path);
        }
    } else {
        std::cout << "Unable to open " << images_file << std::endl;
        exit(-1);
    }
    // 获得样本数量
    int N = image_paths.size();

    Tensor X(N, 28*28);
    Tensor Y(N, 10);

    for (int i = 0; i < N; i++) {
        auto image_path = image_paths[i];
        // 读取图片数据, 将它转换成灰度图 (28,28,1)
        Image im = imread(image_path).gray();
        for (int j = 0; j < X.cols; j++) {
            X[i][j] = im.data[j] / 255.f;   // 像素归一化
        }
        // 倒数第 5 个为该图片的标签，如 mnist/train/039210-num2.png 标签为 2
        char label = image_path[image_path.length()-5];
        // label 转 int 时返回的是 ascii 码，所以需要减 48
        int l = static_cast<int>(label) - 48;
        Y[i][l] = 1.f;  // one hot 编码
    }
    return Data(X, Y);
}
