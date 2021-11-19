/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : main.cc
*   Author      : YunYang1994
*   Created date: 2021-08-10 14:44:01
*   Description :
*
*===============================================================*/

#include "sort.h"

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

 void LoadDetection(std::string file_name, std::map<int, std::vector<DetectionBox>> &detection_boxes)
{
    std::ifstream stream;
    stream.open(file_name);

    if(!stream.is_open())
    {
        std::cerr << "Error: can not find file " << file_name << std::endl;
        assert(0);
    }

    std::string line;
    while(std::getline(stream, line))
    {
        std::istringstream ss;
        char ch;
        int frame;
        float xmin, ymin, xmax, ymax, score;

        DetectionBox det;

        ss.str(line);
        ss >> frame >> ch >> xmin >> ch >> ymin >> ch >> xmax >> ch >> ymax >> ch >> score;
        ss.str("");

        det.box = cv::Rect_<float>(cv::Point_<float>(xmin, ymin), cv::Point_<float>(xmax, ymax));
        det.score = score;

        detection_boxes[frame].push_back(det);
    }
    stream.close();
}


int main()
{
    int cnum = 20;
    cv::RNG rng(0xFFFFFFFF);                                // 生成随机颜色
    cv::Scalar_<int> randColor[cnum];
	for (int i = 0; i < cnum; i++)
		rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

    std::map<int, std::vector<DetectionBox>> detections;
    LoadDetection("data/det.txt", detections);                  // 读取检测框

    cv::Mat frame;
    cv::VideoCapture capture("data/PETS09-S2L1-raw.mp4");        // 读取视频

    TrackingSession *sess = CreateSession(2, 3, 0.3);            // 创建追踪会话

    int frame_id = 0;
    if(!capture.isOpened())
    {
        printf("can not open ...\n");
        return -1;
    }

    while (capture.read(frame))
    {
        auto dets = detections[frame_id];
        auto trks = sess->Update(dets);

        // for(auto &det : dets)
            // cv::rectangle(frame, det.box, randColor[20], 2, 8, 0);

        for(auto &trk : trks)
            cv::rectangle(frame, trk.box, randColor[trk.id % cnum], 2, 8, 0);

        frame_id += 1;
        cv::imshow("output", frame);
        if(char(cv::waitKey(60)) == 'q') break;
    }

    ReleaseSession(&sess);                                  // 结束会话，释放内存
    capture.release();
    return 0;
}


