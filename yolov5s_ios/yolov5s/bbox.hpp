//
//  bbox.hpp
//  detection
//
//  Created by yang on 2020/12/3.
//

#ifndef bbox_hpp
#define bbox_hpp

#include <vector>
#include <opencv2/core/core.hpp>

typedef struct _Bbox {
    int m_xmin;
    int m_ymin;
    int m_xmax;
    int m_ymax;
    int m_label;
    float m_score;

    _Bbox(int xmin, int ymin, int xmax, int ymax, int label, float score):
        m_xmin(xmin), m_ymin(ymin), m_xmax(xmax), m_ymax(ymax), m_label(label), m_score(score) {
    };
} Bbox;

float iou(Bbox box1, Bbox box2);

std::vector<Bbox> nms(std::vector<Bbox> &boxes, float iou_thresh);

void scaleCoords(std::vector<Bbox> &boxes, int org_w, int org_h, int margin=0);

void drawBoxes(cv::Mat image, std::vector<Bbox> bboxes);

#endif /* bbox_hpp */
