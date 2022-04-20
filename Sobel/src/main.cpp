
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


/**
 * Helper struct for convolve function
 */
template<typename T>
struct Vec2D{
	T x, y;
};

struct RGB{
	unsigned char r,g,b;
};

/**
 * Convolve an RGB frame with a floating point matrix.
 *
 * @param frame the RGB frame
 * @param frame_dim Dimensions of the frame {X, Y}
 * @param mat the Matrix
 * @param mat_dim Dimensions of the matrix {X, Y}
 * @param mat_off Offset of the Matrix relative to the pixel to calculate.
 * For normal 3x3 Matrices this should be {-1, -1}
 * @return void. This function writes the result back into frame.
 */

void convolve(unsigned char* frame, Vec2D<size_t> frame_dim, float* mat, Vec2D<size_t> mat_dim, Vec2D<int> mat_off) {
	const size_t buf_width = frame_dim.x + mat_dim.x - 1;
	const size_t buf_height = frame_dim.y + mat_dim.y - 1;
	RGB* frame_buf = (RGB*)frame;
	RGB* buf = (RGB*)calloc(buf_width * buf_height, sizeof(RGB));
	// Move data to new buffer with 0s at edges.
	for (size_t i = 0; i < frame_dim.y; i++)
	{
		size_t dst_idx = (i - mat_off.y) * buf_width - mat_off.x;
		size_t src_idx = i * frame_dim.x;
		memcpy(&buf[dst_idx], &frame_buf[src_idx], frame_dim.x * sizeof(RGB));
	}
	// Convolve
	for (size_t y = 0; y < frame_dim.y; y++)
	{
		for (size_t x = 0; x < frame_dim.x; x++)
		{
			float r = 0.0f, g = 0.0f, b = 0.0f;
			for (size_t v = 0; v < mat_dim.y; v++)
			{
				for (size_t u = 0; u < mat_dim.x; u++)
				{
					float mat_val = mat[mat_dim.x * v + u];
					size_t buf_idx = (y + v) * buf_width + (x + u);
					r += mat_val * buf[buf_idx].r;
					g += mat_val * buf[buf_idx].g;
					b += mat_val * buf[buf_idx].b;
				}
			}
			size_t frame_idx = y * frame_dim.x + x;
			frame_buf[frame_idx].r = (unsigned char)r;
			frame_buf[frame_idx].g = (unsigned char)g;
			frame_buf[frame_idx].b = (unsigned char)b;
		}
	}
	free(buf);
}


void sobel(unsigned char * frame){

}


void blur(unsigned char * frame){

}


