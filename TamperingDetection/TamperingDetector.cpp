/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : TamperingDetector.cpp
*   Author      : YunYang1994
*   Created date: 2021-09-10 10:20:52
*   Description :
*
*===============================================================*/

#include "TamperingDetector.hpp"

TamperingDetector::TamperingDetector(int mixture, float tampering_ratio):
    m_mixture(mixture), m_tampering_ratio(tampering_ratio) {
        m_bg = cv::createBackgroundSubtractorMOG2();
        m_bg->setNMixtures(m_mixture);
    }
bool TamperingDetector::isTampering(const cv::Mat &frame, int min_size,
        std::vector<std::vector<cv::Point>> &contours) {
    cv::Mat fore, back, gray_frame;

    cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
    int width = gray_frame.rows;
    int height = gray_frame.cols;

    m_bg->apply(gray_frame, fore);
    m_bg->getBackgroundImage(back);

    cv::erode(fore, fore, cv::Mat());
    // cv::erode(fore, fore, cv::Mat());
    cv::dilate(fore, fore, cv::Mat());
    // cv::dilate(fore, fore, cv::Mat());
    // cv::dilate(fore, fore, cv::Mat());

    cv::findContours(fore, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    int tampering_area = 0;
    int frame_area = width * height;

    std::vector<cv::Rect> boundRect(contours.size());
    for( int i = 0; i < contours.size(); i++ ) {
        boundRect[i] = boundingRect( contours[i] );
    }

    for( int i = 0; i< contours.size(); i++ ) {
        // eliminates small boxes
        if(boundRect[i].width>=min_size || boundRect[i].height>=min_size)
            tampering_area= tampering_area + (boundRect[i].height)*(boundRect[i].width);

        if(tampering_area >= frame_area*m_tampering_ratio)
            return true;
    }
    return false;
}
