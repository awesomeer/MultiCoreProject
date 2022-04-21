
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>

#include "../inc/kernel.h"
#include <iostream>
using namespace std;
using namespace cv;

#include <chrono>
#include <vector>

/*
GPU
GREY - 		552.5
SOBEL - 	202.7
GAUSSIAN - 	160.034

CPU
GREY - 		186
SOBEL - 	7.3
GAUSSIAN - 	15

*/


int main(int argc, char** argv) {

	VideoCapture camera;

	if (argc > 1)
		camera.open(argv[1]);
	else
		camera.open(1);

	if (!camera.isOpened())
		return -1;
	camera.set(CAP_PROP_FRAME_WIDTH, 1920);
	camera.set(CAP_PROP_FRAME_HEIGHT, 1080);

	//cout << camera.get(CAP_PROP_FRAME_WIDTH) << "x" << camera.get(CAP_PROP_FRAME_HEIGHT) << endl;

	Mat cap;
	initCuda();
	namedWindow("Video Stream");
	resizeWindow("Video Stream", 1920, 1080);
	
	
	//VideoWriter video("outcpp.mp4",VideoWriter::fourcc('m','p','4','v'),30, Size(1920,1080));
	
	vector<double> fps;

	chrono::system_clock::time_point start, end;
	chrono::duration<double> time;
	FilterType filtertype = GAUSSIAN;
	bool stoploop = true;
	
	Mat blur;
	while (stoploop) {
		camera >> cap;
		if (cap.empty())
			break;
		
		start = chrono::system_clock::now();
		filter(cap.data, filtertype);
		end = chrono::system_clock::now();

		time = end - start;
		cout << 1 / time.count() << endl;
		fps.push_back(1.0/time.count());

		float fps = 1.0f / time.count();
        cv::String fps_text = std::to_string(fps).substr(0, 5) +" FPS";
        cv::Point org = cv::Point(cap.cols - 200, 30);
        int font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        cv::Scalar color = { 0, 255, 0 };
        cv::putText(cap, fps_text, org, font, font_scale, color);
		imshow("Video Stream", cap);
		//video.write(cap);

		switch(waitKey(10)){
			case 27:{
				stoploop = false;
				break;
			}
			case 49:{
				filtertype = GREY;
				break;
			}
			case 50:{
				filtertype = SOBEL;
				break;
			}
			case 51:{
				filtertype = GAUSSIAN;
				break;
			}
		}
	}
	cout << cap.size << endl;

	double sum = 0;
	for(int i = 0; i < fps.size(); i++){
		sum += fps[i];
	}
	cout << "Average FPS: " << sum/fps.size() << endl;

	//video.release();
	cap.release();
	freeCuda();

	return 0;
}