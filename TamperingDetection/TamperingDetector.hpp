/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : TamperingDetector.hpp
*   Author      : YunYang1994
*   Created date: 2021-09-09 20:36:35
*   Description :
*
*===============================================================*/

#ifndef _TAMPERINGDETECTOR_H
#define _TAMPERINGDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

class TamperingDetector {
public:
    TamperingDetector(int mixture, float tampering_ratio);
    bool isTampering(const cv::Mat &frame, int min_size, std::vector<std::vector<cv::Point>> &contours);

private:
    int m_mixture;
    float m_tampering_ratio;
    cv::Ptr<cv::BackgroundSubtractorMOG2> m_bg;
};

#ifdef __cplusplus
}
#endif
#endif //TAMPERINGDETECTOR_H

