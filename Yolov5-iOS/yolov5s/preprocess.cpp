//
//  preprocess.cpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#include "preprocess.hpp"

void preprocess(const cv::Mat &image, float *data) {
    assert((image.channels() == 3) && (data != nullptr));           // input must be a clor image and data is not nullptr
    
    int height = 208, width = 208, channels = 12;
    auto temp = (float*)malloc((height*width*channels) * sizeof(float));
    
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(416, 416));
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
    resize_image.convertTo(resize_image, CV_32F);                   // CV_8U -> CV_32F
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = width * i + j;
                                                                    // split and concat image
            float* c1 = resize_image.ptr<float>(2*i, 2*j);
            memcpy(&temp[k*12+0], c1, 3*sizeof(float));

            float* c2 = resize_image.ptr<float>(2*i+1, 2*j);
            memcpy(&temp[k*12+3], c2, 3*sizeof(float));

            float* c3 = resize_image.ptr<float>(2*i, 2*j+1);
            memcpy(&temp[k*12+6], c3, 3*sizeof(float));

            float* c4 = resize_image.ptr<float>(2*i+1,2*j+1);
            memcpy(&temp[k*12+9], c4, 3*sizeof(float));
        }
    }
                                                                   // HWC to CHW and Normalize
    for (int c = 0; c <channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int dst_idx = c * height * width + h * width + w;
                int src_idx = h * width * channels + w * channels + c;
                data[dst_idx] = temp[src_idx] / 255.f;
            }
        }
    }
    
    free(temp);
}

