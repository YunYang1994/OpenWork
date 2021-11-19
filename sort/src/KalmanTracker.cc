/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : KalmanTracker.cc
*   Author      : YunYang1994
*   Created date: 2021-07-25 11:12:08
*   Description :
*
*===============================================================*/

#include "KalmanTracker.h"


int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void KalmanTracker::InitKf(StateType state_mat)
{
	int state_num = 7;
	int measure_num = 4;

	kf = cv::KalmanFilter(state_num, measure_num, 0);
	measurement = cv::Mat::zeros(measure_num, 1, CV_32F);

	kf.transitionMatrix = (cv::Mat_<float>(state_num, state_num) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, cv::Scalar::all(1));

	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = state_mat.x + state_mat.width / 2;
	kf.statePost.at<float>(1, 0) = state_mat.y + state_mat.height / 2;
	kf.statePost.at<float>(2, 0) = state_mat.area();
	kf.statePost.at<float>(3, 0) = state_mat.width / state_mat.height;
}


// Predict the estimated bounding box.
StateType KalmanTracker::Predict()
{
	// predict
    cv::Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predict_box = convert_x_to_bbox(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predict_box);
	return m_history.back();
}


// Update the state vector with observed bounding box.
void KalmanTracker::Update(StateType state_mat)
{
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	measurement.at<float>(0, 0) = state_mat.x + state_mat.width / 2;
	measurement.at<float>(1, 0) = state_mat.y + state_mat.height / 2;
	measurement.at<float>(2, 0) = state_mat.area();
	measurement.at<float>(3, 0) = state_mat.width / state_mat.height;

	// update
	kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::GetState()
{
    cv::Mat s = kf.statePost;
	return convert_x_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::convert_x_to_bbox(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}


