
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>

//#include "../inc/kernel.h"
#include <iostream>
using namespace std;
using namespace cv;

#include <vector>
#include <chrono>

#define WIDTH 1280
#define HEIGHT 720

void greyScale(unsigned char * frame);
void sobel(unsigned char * frame);
void blur(unsigned char * frame);


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
	vector<float> times;

	namedWindow("Video Stream");
	resizeWindow("Video Stream", 1280, 720);
	
	chrono::system_clock::time_point start, end;
	chrono::duration<double> time;

	while (true) {
		camera >> cap;
		if (cap.empty())
			break;
		
		start = chrono::system_clock::now();
		greyScale(cap.data);
		end = chrono::system_clock::now();
		time = end - start;
		cout << 1 / time.count() << endl;
		times.push_back(time.count());
		
		imshow("Video Stream", cap);

		if (waitKey(10) == 27)
			break;
	}

	unsigned char* data = cap.data;
	cout << cap.size << endl;

	double frames = 0;
	for(int i = 0; i < times.size(); i++){
		frames += (double) times[i];
	}
	frames /= times.size();
	cout << "Average FPS: " << 1/frames << endl;

	return 0;
}


void greyScale(unsigned char * frame){
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			int index = 3*(c + r*WIDTH);
			int sum = frame[index] + frame[index+1] + frame[index+2];
			sum /= 3;
			frame[index] = sum;
			frame[index+1] = sum;
			frame[index+2] = sum;
		}
	}

}


void sobel(unsigned char * frame){

}


void blur(unsigned char * frame){

}


