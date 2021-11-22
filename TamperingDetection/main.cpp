/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : main.cpp
*   Author      : YunYang1994
*   Created date: 2021-09-09 20:33:54
*   Description :
*
*===============================================================*/

#include <algorithm>
#include <opencv2/opencv.hpp>
#include "TamperingDetector.hpp"

int main() {
    cv::VideoWriter writer;
    writer.open("result.mp4", cv::VideoWriter::fourcc('M', 'P', '4', '2'), 30, cv::Size(960, 540));
    cv::VideoCapture cap(0);

    cv::Mat frame, resize_frame;
    TamperingDetector tamperDet(3, 0.9);

    while(true) {
        cap >> frame;
        if(frame.empty()) break;

        float ratio = 0.5;
        std::vector<std::vector<cv::Point>> contours;

        int width = int(frame.cols*ratio);
        int height = int(frame.rows*ratio);

        cv::resize(frame, resize_frame, cv::Size(width, height));

        int min_size = std::min(width, height) * 0.7;
        std::cout << "height= " << height << " width=" << width << " min_size=" << min_size <<  std::endl;

        if(tamperDet.isTampering(resize_frame, min_size, contours)) {
            cv::putText(resize_frame,"warning", cv::Point(5,30), cv::FONT_HERSHEY_SIMPLEX,1, cv::Scalar(0,0,255),2);
        }

        // cv::drawContours(resize_frame, contours, -1, cv::Scalar(255, 0, 0), 1);

        cv::imshow("result", resize_frame);
        cv::waitKey(10);

        writer.write(resize_frame);
    }
}
