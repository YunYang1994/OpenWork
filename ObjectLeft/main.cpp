/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : AbandonedObjectDetector.cpp
*   Author      : YunYang1994
*   Created date: 2021-10-13 22:35:24
*   Description :
*
*===============================================================*/

#include<ctime>
#include<queue>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class AbandonedObjectDetector {
public:
    cv::Mat diff, back, aban, sub;

    AbandonedObjectDetector () {
        count = 0;
        npixel = 100;
    };

    void init() {
        for (int i=0; i<160; i++)
            for (int j=0; j<160; j++) {
                sum2[i][j] = 0;
                avgq2[i][j] = 0;
            }

        flag = 0;
        aban = cv::Mat(200,200,CV_8UC1,cv::Scalar(0));
    }

    void update(const cv::Mat &frame) {
        cv::Rect roi(0, 65, frame.cols, frame.rows-65);
        auto frame1 = frame.clone();
        auto image = frame1(roi);
        cv::resize(image, image, cv::Size(160,160));

        count++;
        cv::Mat gray;
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        diff=gray.clone();
        cv::absdiff(gray, gray, diff);

        back=gray.clone();
        cv::absdiff(back,back,back);

        for (int i=0; i<160; i++)
            for (int j=0; j<160; j++) {
                col2[i][j] = gray.at<uchar>(cv::Point(i,j));  //save latest value
                colq2[i][j].push(col2[i][j]);  //push latest value

                if (count<npixel) {
                    sum2[i][j] = sum2[i][j]+col2[i][j];
                    avgq2[i][j] = 0;	//avg zero till queue is full
                } else {                //Background acquisition complete
                    colq2[i][j].pop();  //take latest pixel
                    sum2[i][j] = sum2[i][j]+col2[i][j]-colq2[i][j].front(); //take sum of latest 150 pixels
                    avgq2[i][j] = sum2[i][j]/npixel;
                }

                back.at<uchar>(cv::Point(i,j)) = avgq2[i][j]; //get averaged background
                if (col2[i][j]-avgq2[i][j]>10 || col2[i][j]-avgq2[i][j]<(-10)) {
                    diff.at<uchar>(cv::Point(i,j))=col2[i][j];   // BG modelling step
                }
            }

        if (count<npixel) {
            std::cout << "Please wait, acquiring Background ..." << std::endl;
        }

        int interval;
        if (flag==0) {
            cv::absdiff(back,back,aban);//initialize aban once only
            interval=300;
            flag=10;
        }//initial block in verilog, with delay

        if(flag==10 && count>=npixel) {         //this loop runs once
			aban=back.clone();//extra loop run to ensure stable initial background
			flag =20;
		}

        bool reset_background = false;
        if(reset_background)
        {
            std::cout<<"New Interval \n";
            aban=back.clone(); //not aban=back as they'll become pointers pointing to same address
        }

        cv::absdiff(back,aban,sub);

        cv::resize(diff,diff,cv::Size(400,400));
        cv::resize(back,back,cv::Size(400,400));
        cv::resize(image,image,cv::Size(400,400));
        cv::resize(sub,sub,cv::Size(400,400));

        imshow("img",image);
		imshow("back",back);
        imshow("aban", aban);
		imshow("Abandoned Objects",sub);
        cv::waitKey(1);
    };

private:
    int count;
    int npixel;
    int flag;
    std::queue<int>colq;
    std::queue<int>colq2[160][160];

    clock_t clk;
    int col2[160][160];
    int sum2[160][160];
    float avgq2[160][160];
};

int main () {
    cv::Mat frame;
    cv::VideoCapture cap("ObjectLeft.mp4");

    auto detector = AbandonedObjectDetector();
    detector.init();

    while(true) {
        cap >> frame;
        if(frame.empty()) break;
        detector.update(frame);
    }
}
