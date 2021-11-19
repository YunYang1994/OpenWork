//
//  postprocess.hpp
//  detection
//
//  Created by yang on 2020/12/4.
//

#ifndef postprocess_hpp
#define postprocess_hpp

#include "bbox.hpp"

std::vector<Bbox> postprocess(const std::vector<float *> &feature_map_ptrs, float conf_thresh, float iou_thresh);

#endif /* postprocess_hpp */
