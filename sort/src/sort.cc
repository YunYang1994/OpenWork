/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : sort.cc
*   Author      : YunYang1994
*   Created date: 2021-08-11 20:31:32
*   Description :
*
*===============================================================*/


#include "sort.h"
#include "Hungarian.h"
#include "KalmanTracker.h"

#include <atomic>
#include <vector>
#include <cfloat> // for DBL_MAX
#include <iomanip>    // to format image names using setw() and setfill()
#include <unistd.h>   // to check file existence using POSIX function access(). On Linux include <unistd.h>.


class Sort: public TrackingSession
{
public:
    Sort(int max_age, int min_hits, float iou_threshold):
        m_max_age(max_age), m_min_hits(min_hits), m_iou_threshold(iou_threshold)
    {
        m_frame_count = 0;
        m_trackers = {};
        ms_num_session++;
    }
    std::vector<TrackingBox> Update(const std::vector<DetectionBox> &dets) override;
    ~Sort() { ms_num_session--; };

private:
    float m_iou_threshold;
    int m_max_age, m_min_hits, m_frame_count;
    std::vector<KalmanTracker> m_trackers;

    static std::atomic<int> ms_num_session;
};

std::atomic<int> Sort::ms_num_session(0);

TrackingSession *CreateSession(int max_age, int min_hits, float iou_threshold) {
    return new Sort(max_age, min_hits, iou_threshold);
}


void ReleaseSession(TrackingSession **session_ptr) {
    if (session_ptr && *session_ptr) {
        delete *session_ptr;
        session_ptr = nullptr;
    }
}


static double compute_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float intersection_area = (bb_test & bb_gt).area();
    float union_area = bb_test.area() + bb_gt.area() - intersection_area;

    if (union_area < DBL_EPSILON)
        return 0;
    return (double)(intersection_area / union_area);
}


static void AssociateDetectionsToTrackers(const std::vector<DetectionBox> &dets, const std::vector<TrackingBox> &trks, float iou_threshold,
        std::vector<std::vector<int>> &matches, std::vector<int> &unmatched_detections, std::vector<int> &unmatched_trackers)
{
    int det_num = dets.size();
    int trk_num = trks.size();

    if(trk_num==0)
    {
        for(int i=0; i<det_num; i++)
            unmatched_detections.push_back(i);
        return;
    }

    std::vector<std::vector<double>> iou_matrix;
    iou_matrix .resize(det_num, vector<double>(trk_num, 0));

    for(int i=0; i<det_num; i++)
        for(int j=0; j<trk_num; j++)
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iou_matrix[i][j] = 1 - compute_iou(dets[i].box, trks[j].box);

    // solve the assignment problem using hungarian algorithm.
    HungarianAlgorithm hungalgo;
    std::vector<int> assignment;

    // the resulting assignment is [detection : tracker], with len=preNum
    hungalgo.Solve(iou_matrix, assignment);

    for(int i=0; i<det_num; i++)
    {
        int j = assignment[i];
        // unassigned label will be set as -1 in the assignment algorithm
        if((j != -1) && (1-iou_matrix[i][j] >= iou_threshold))
        {
            std::vector<int> match = {i, j};
            matches.push_back(match);
        }
        else
            unmatched_detections.push_back(i);
    }

    for(int i=0; i<trk_num; i++)
        for(int j=0; j<matches.size(); j++)
            if(i != matches[j][1]) unmatched_trackers.push_back(i);
}


std::vector<TrackingBox> Sort::Update(const std::vector<DetectionBox> &dets)
{
    m_frame_count += 1;
    std::vector<TrackingBox> trks;

    for(auto it = m_trackers.begin(); it != m_trackers.end();)
    {
        TrackingBox trk;
        trk.box = it->Predict();

        if(trk.box.x >= 0 && trk.box.y >=0)
        {
            trks.push_back(trk);
            it++;
        }
        else
        {
            it = m_trackers.erase(it);
        }
    }

    std::vector<std::vector<int>> matches;
    std::vector<int> unmatched_detections, unmatched_trackers;

    AssociateDetectionsToTrackers(dets, trks, m_iou_threshold, matches, unmatched_detections, unmatched_trackers);

    // update matched trackers with assigned detections.
    for(auto &m : matches)
        m_trackers[m[1]].Update(dets[m[0]].box);

    // create and initialise new trackers for unmatched detections
    for(auto &d : unmatched_detections)
    {
        KalmanTracker tracker = KalmanTracker(dets[d].box);
        m_trackers.push_back(tracker);
    }

    trks.clear();
    // get trackers' output
    for(auto it = m_trackers.begin(); it != m_trackers.end(); it++)
    {
        if(((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= m_min_hits || m_frame_count <= m_min_hits))
        {
            TrackingBox trk;
            trk.box = it->GetState();
            trk.id = it->m_id + 1;
            trks.push_back(trk);
        }

        // remove dead tracklet
        if((*it).m_time_since_update > m_max_age)
        {
            it = m_trackers.erase(it);
            it--;
        }
    }
    return trks;
}


