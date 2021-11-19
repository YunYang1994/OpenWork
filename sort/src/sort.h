/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : sort.h
*   Author      : YunYang1994
*   Created date: 2021-08-10 14:59:53
*   Description :
*
*===============================================================*/

#pragma once
#include <opencv2/core.hpp>

typedef struct DetectionBox
{
    float score;
    cv::Rect_<float> box;
}DetectionBox;

typedef struct TrackingBox
{
    int id;
    cv::Rect_<float> box;
}TrackingBox;


class TrackingSession {
    public:
        virtual ~TrackingSession() {};
        virtual std::vector<TrackingBox> Update(const std::vector<DetectionBox> &dets) = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

TrackingSession *CreateSession(int max_age, int min_hits, float iou_threshold);
void  ReleaseSession(TrackingSession **session_ptr);

#ifdef __cplusplus
}
#endif


