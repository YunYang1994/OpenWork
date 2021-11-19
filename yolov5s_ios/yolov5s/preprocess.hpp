//
//  preprocess.hpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#ifndef preprocess_hpp
#define preprocess_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void preprocess(const cv::Mat &image, float *data);

#endif /* preprocess_hpp */
