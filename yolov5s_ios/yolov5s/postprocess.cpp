//
//  postprocess.cpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#include "postprocess.hpp"

typedef enum _stride {
    SMALL  = 8,
    MEDIUM = 16,
    LARGE  = 32,
} stride;

typedef struct _Anchor {
    int m_width;
    int m_height;

    _Anchor(int width, int height): m_width(width), m_height(height) { };
} Anchor;

static const std::vector<Anchor> anchor_s = {Anchor(10,  13), Anchor(16,   30), Anchor(33,   23)};
static const std::vector<Anchor> anchor_m = {Anchor(30,  61), Anchor(62,   45), Anchor(59,  119)};
static const std::vector<Anchor> anchor_l = {Anchor(116, 90), Anchor(156, 198), Anchor(373, 326)};

void boxesRegression(const float *feature_map_ptr, std::vector<Bbox>& boxes, std::vector<Anchor> anchor, float conf_thresh, stride s) {
    int nc = 3, na = 85;
    int nx = 0, ny = 0;
    
    switch (s) {
        case SMALL:
            nx = ny = 52;
            break;
        
        case MEDIUM:
            nx = ny = 26;
            break;
        
        case LARGE:
            nx = ny = 13;
            break;
            
        default:
            break;
    }
    
    auto getFeatureMapValue = [na, ny, nx, feature_map_ptr](int i, int j, int k, int l) {
        return *(feature_map_ptr + na * nx * ny * k + nx * ny * l + nx * i + j);
    };
    
    for (int k = 0; k < nc; k++) {
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                float conf = getFeatureMapValue(i, j, k, 4);
                if (conf > conf_thresh) {                                                  // filter some boxes of low score
                    float x = (getFeatureMapValue(i, j, k, 0) * 2.f - 0.5f + j) * s;
                    float y = (getFeatureMapValue(i, j, k, 1) * 2.f - 0.5f + i) * s;

                    float w = pow((getFeatureMapValue(i, j, k, 2) * 2.f), 2.f) * anchor[k].m_width;
                    float h = pow((getFeatureMapValue(i, j, k, 3) * 2.f), 2.f) * anchor[k].m_height;

                    int xmin = x - 0.5f * w;
                    int ymin = y - 0.5f * h;
                    int xmax = x + 0.5f * w;
                    int ymax = y + 0.5f * h;
                    
                    int label;
                    float max_class_prob = 0.f;
                    
                    for (int c = 5; c < na; c++) {                                        // argmax class probability
                        float class_prob = getFeatureMapValue(i, j, k, c);
                        if (class_prob > max_class_prob) {
                            max_class_prob = class_prob;
                            label = c - 5;
                        }
                    }
                    
                    float score = conf * max_class_prob;
                    boxes.emplace_back(xmin, ymin, xmax, ymax, label, score);
                }
            }
        }
    }
}

std::vector<Bbox> postprocess(const std::vector<float *> &feature_map_ptrs, float conf_thresh, float iou_thresh) {
    std::vector<Bbox> boxes;
    
    boxesRegression(feature_map_ptrs[0], boxes, anchor_s, conf_thresh, SMALL);
    boxesRegression(feature_map_ptrs[1], boxes, anchor_m, conf_thresh, MEDIUM);
    boxesRegression(feature_map_ptrs[2], boxes, anchor_l, conf_thresh, LARGE);
    
    auto picked_boxes = nms(boxes, iou_thresh);
    return picked_boxes;
}

