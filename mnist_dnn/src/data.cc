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

#include <stdlib.h>
#include "data.h"

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


