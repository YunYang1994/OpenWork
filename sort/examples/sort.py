#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : sort.py
#   Author      : YunYang1994
#   Created date: 2021-08-09 20:22:38
#   Description :
#
#================================================================

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def load_detection(det_text):
    """
    return dets - a dict of detections in the format
        {
            frame_id: [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        }
    """
    dets = {}
    lines = open(det_text, "r").readlines()
    for line in lines:
        line = line.split(",")
        frame_id = int(line[0])
        xmin = int(line[1])
        ymin = int(line[2])
        xmax = int(line[3])
        ymax = int(line[4])
        score = float(line[5])
        bbox = [xmin, ymin, xmax, ymax, score]
        if frame_id not in dets.keys():
            dets[frame_id] = []
        dets[frame_id].append(bbox)
    return dets


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou(bb_gt, bb_test):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return(o)


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):            # 将 bbox 从 [x1,y1,x2,y2] 格式变成 [x,y,s,r] 格式
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):     # 将 bbox 从 [x,y,s,r] 格式变成 [x1,y1,x2,y2] 格式
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],      # 7x7 维度
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],      # 4x7 维度
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(dets, trks, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
        dets:
            [[x1,y1,x2,y2,score],...]
        trks:
            [[x1,y1,x2,y2,tracking_id],...]
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if(len(trks)==0):
        return np.empty(0,dtype=int), np.arange(len(dets),dtype=int), np.empty(0,dtype=int)

    # 计算目标检测的 bbox 和卡尔曼滤波的 bbox 之间的 iou 矩阵
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d,t] = iou(det, trk)

    # 计算匈牙利算法匹配的 matched_indices: [[d,t] ...]
    matched_indices = linear_assignment(-iou_matrix)

    # 没有匹配上的目标检测 bbox 放入 unmatched_detections 列表
    # 表示有新物体进入画面了，后面要新增跟踪器来追踪新物体
    unmatched_detections = []
    for d, det in enumerate(dets):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    # 没有匹配上的卡尔曼滤波 bbox 放入 unmatched_trackers 列表
    # 表示之前跟踪的物体消失了，后面要删除对应的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trks):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:   # 遍历 matched_indices 矩阵
        if(iou_matrix[m[0], m[1]]<iou_threshold):   # 将 iou 值小于 iou_threshold 的匹配结果分别放入
            unmatched_detections.append(m[0])       # unmatched_detections 和 unmatched_trackers 列表中
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))          # 匹配上的则以 [[d,t]...] 形式放入 matches 矩阵

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    # 返回：跟踪成功的矩阵，新增物体的矩阵， 消失物体的矩阵
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age      # 在没有目标检测关联的情况下追踪器存活的最大帧数
        self.min_hits = min_hits    # 追踪器初始化前的最小关联检测数量
        self.iou_threshold = iou_threshold

        self.trackers = []          # 用于存储卡尔曼滤波追踪器的列表
        self.frame_count = 0        # 当前追踪帧的编号

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        trks = []       # 用于存放跟踪预测的 bbox: [x1,y1,x2,y2,id]

        for i, tracker in enumerate(self.trackers):       # 遍历卡尔曼跟踪列表
            pos = tracker.predict()[0]                    # 用卡尔曼跟踪器 t 预测 bbox

            if not np.any(np.isnan(pos)):                 # 如果卡尔曼的预测框有效
                trks.append([pos[0], pos[1], pos[2], pos[3], 0])    # 存放上一帧所有物体预测有效的 bbox
            else:
                self.trackers.remove(tracker)             # 如果无效， 删除该滤波器

        trks = np.array(trks)
        self.trks = trks                # 为了显示跟踪器预测的框，把它拿出来

        # 将目标检测的 bbox 和卡尔曼滤波预测的跟踪 bbox 匹配
        # 获得 跟踪成功的矩阵，新增物体的矩阵，消失物体的矩阵
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 跟踪成功的物体 bbox 信息更新到对应的卡尔曼滤波器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 为新增物体创建新的卡尔曼滤波跟踪器
        for i in unmatched_dets:
            tracker = KalmanBoxTracker(dets[i,:])
            self.trackers.append(tracker)

        # 跟踪器更新校正后，输出最新的 bbox 和 id
        ret = []
        for tracker in self.trackers:
            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = tracker.get_state()[0]
                ret.append([d[0], d[1], d[2], d[3], tracker.id+1]) # +1 as MOT benchmark requires positive

            # 长时间离开画面/跟踪失败的物体从卡尔曼跟踪器列表中删除
            if(tracker.time_since_update > self.max_age):
                self.trackers.remove(tracker)

        # 返回当前画面中所有被跟踪物体的 bbox 和 id，矩阵形式为 [[x1,y1,x2,y2,id]...]
        return np.array(ret) if len(ret) > 0 else np.empty((0,5))

if __name__ == "__main__":
    np.random.seed(0)
    cnum = 64
    colours = np.random.rand(cnum, 3) * 255       # 假设最大的跟踪ID为 cnum，每个类别一个颜色条
    cap = cv2.VideoCapture("data/PETS09-S2L1-raw.mp4")
    detections = load_detection("data/det.txt")

    mot_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.3)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_id in detections.keys():
            bboxes = detections[frame_id]
            trks = mot_tracker.update(np.array(bboxes))

            for trk in trks:
                trk = trk.astype(int)
                frame = cv2.rectangle(frame, (trk[0], trk[1]), (trk[2], trk[3]), colours[trk[4]%cnum], 2)

            # # 显示卡尔曼跟踪器的预测框
            # for trk in mot_tracker.trks:
                # trk = trk.astype(int)
                # frame = cv2.rectangle(frame, (trk[0], trk[1]), (trk[2], trk[3]), [255, 0, 0], 2)

        cv2.imshow("tracking", frame)
        print(frame_id, ": ", trks)

        frame_id += 1
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

