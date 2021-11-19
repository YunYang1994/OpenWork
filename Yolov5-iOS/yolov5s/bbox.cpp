//
//  bbox.cpp
//  detection
//
//  Created by yang on 2020/12/3.
//

#include "bbox.hpp"
#include <string>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.m_xmax - box1.m_xmin + 1) * (box1.m_ymax - box1.m_ymin + 1);
    float area2 = (box2.m_xmax - box2.m_xmin + 1) * (box2.m_ymax - box2.m_ymin + 1);

    float x11 = std::max(box1.m_xmin, box2.m_xmin);
    float y11 = std::max(box1.m_ymin, box2.m_ymin);
    float x22 = std::min(box1.m_xmax, box2.m_xmax);
    float y22 = std::min(box1.m_ymax, box2.m_ymax);

    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);
    return intersection / (area1 + area2 - intersection);
}

std::vector<Bbox> nms(std::vector<Bbox> &boxes, float iou_thresh) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
        return box1.m_score < box2.m_score;                         // 升序排列, 令 score 最大的 box 在 vector 末端
    };

    std::sort(boxes.begin(), boxes.end(), cmpScore);
    std::vector<Bbox> picked_boxes;

    while (boxes.size() > 0) {
        auto box = boxes.back();
        picked_boxes.emplace_back(box);

        boxes.pop_back();
        for(auto iter = boxes.begin(); iter != boxes.end();) {
            if (iou(box, *iter) >= iou_thresh) {
                boxes.erase(iter);
            } else {
                iter++;
            }
        }
    }
    return picked_boxes;
}

static inline float clip(float val, float min, float max) {
    return (val < min) ? min: (val > max) ? max: val;
}

void scaleCoords(std::vector<Bbox>& boxes, int org_w, int org_h, int margin) {
    float scale_w = 416.f / org_w;
    float scale_h = 416.f / org_h;

    for(int i=0; i<boxes.size(); i++){
        boxes[i].m_xmin = clip((boxes[i].m_xmin / scale_w - margin), 0.f, org_w-1.);
        boxes[i].m_ymin = clip((boxes[i].m_ymin / scale_h - margin), 0.f, org_h-1.);
        boxes[i].m_xmax = clip((boxes[i].m_xmax / scale_w + margin), 0.f, org_w-1.);
        boxes[i].m_ymax = clip((boxes[i].m_ymax / scale_h + margin), 0.f, org_h-1.);
    }
}

static std::vector<cv::Scalar> COLOR = {
    cv::Scalar(255,   0,   0), cv::Scalar(255,  19,   0), cv::Scalar(255, 38,    0), cv::Scalar(255,  57,   0),
    cv::Scalar(255,  76,   0), cv::Scalar(255,  95,   0), cv::Scalar(255, 114,   0), cv::Scalar(255, 133,   0),
    cv::Scalar(255, 153,   0), cv::Scalar(255, 172,   0), cv::Scalar(255, 191,   0), cv::Scalar(255, 210,   0),
    cv::Scalar(255, 229,   0), cv::Scalar(255, 248,   0), cv::Scalar(242, 255,   0), cv::Scalar(223, 255,   0),
    cv::Scalar(203, 255,   0), cv::Scalar(184, 255,   0), cv::Scalar(165, 255,   0), cv::Scalar(146, 255,   0),
    cv::Scalar(127, 255,   0), cv::Scalar(108, 255,   0), cv::Scalar( 89, 255,   0), cv::Scalar( 70, 255,   0),
    cv::Scalar( 51, 255,   0), cv::Scalar( 31, 255,   0), cv::Scalar( 12, 255,   0), cv::Scalar(  0, 255,   6),
    cv::Scalar(  0, 255,  25), cv::Scalar(  0, 255,  44), cv::Scalar(  0, 255,  63), cv::Scalar(  0, 255,  82),
    cv::Scalar(  0, 255, 102), cv::Scalar(  0, 255, 121), cv::Scalar(  0, 255, 140), cv::Scalar(  0, 255, 159),
    cv::Scalar(  0, 255, 178), cv::Scalar(  0, 255, 197), cv::Scalar(  0, 255, 216), cv::Scalar(  0, 255, 235),
    cv::Scalar(  0, 255, 255), cv::Scalar(  0, 225, 255), cv::Scalar(  0, 216, 255), cv::Scalar(  0, 197, 255),
    cv::Scalar(  0, 178, 255), cv::Scalar(  0, 159, 255), cv::Scalar(  0, 140, 255), cv::Scalar(  0, 121, 255),
    cv::Scalar(  0, 102, 255), cv::Scalar(  0,  82, 255), cv::Scalar(  0,  63, 255), cv::Scalar(  0,  44, 255),
    cv::Scalar(  0,  25, 255), cv::Scalar(  0,   6, 255), cv::Scalar( 12,   0, 255), cv::Scalar( 31,   0, 255),
    cv::Scalar( 50,   0, 255), cv::Scalar( 70,   0, 255), cv::Scalar( 89,   0, 255), cv::Scalar(108,   0, 255),
    cv::Scalar(127,   0, 255), cv::Scalar(146,   0, 255), cv::Scalar(165,   0, 255), cv::Scalar(184,   0, 255),
    cv::Scalar(204,   0, 255), cv::Scalar(223,   0, 255), cv::Scalar(242,   0, 255), cv::Scalar(255,   0, 248),
    cv::Scalar(255,   0, 229), cv::Scalar(255,   0, 210), cv::Scalar(255,   0, 191), cv::Scalar(255,   0, 172),
    cv::Scalar(255,   0, 152), cv::Scalar(255,   0, 133), cv::Scalar(255,   0, 114), cv::Scalar(255,   0,  95),
    cv::Scalar(255,   0,  76), cv::Scalar(255,   0,  57), cv::Scalar(255,   0,  38), cv::Scalar(255,   0,  19),
};

static std::vector<std::string> CLASS = {                                                                       // COCO
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

void drawBoxes(cv::Mat image, std::vector<Bbox> boxes) {
    for(int i=0; i<boxes.size(); i++) {
        int label = boxes[i].m_label;
        cv::rectangle(image, cv::Point(boxes[i].m_xmin, boxes[i].m_ymin),
                             cv::Point(boxes[i].m_xmax, boxes[i].m_ymax), COLOR[label], 3, 8, 0);
        cv::putText(image, CLASS[label], cv::Point(boxes[i].m_xmin, boxes[i].m_ymin-2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    }
}

