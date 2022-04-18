
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>

#include "../inc/kernel.h"
#include <iostream>
using namespace std;
using namespace cv;

#include <chrono>



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


	Mat cap;
	initCuda();
	namedWindow("Video Stream");
	resizeWindow("Video Stream", 1920, 1080);
	
	chrono::system_clock::time_point start, end;
	chrono::duration<double> time;

	while (true) {
		camera >> cap;
		if (cap.empty())
			break;
		
		start = chrono::system_clock::now();
		filter(cap.data);
		end = chrono::system_clock::now();
		time = end - start;
		cout << 1 / time.count() << endl;
		
		imshow("Video Stream", cap);

		if (waitKey(10) == 27)
			break;
	}
	unsigned char* data = cap.data;
	cout << cap.size << endl;

	freeCuda();

	return 0;
}