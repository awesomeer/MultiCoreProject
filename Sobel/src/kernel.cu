#include "../inc/kernel.h"

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cmath>

#include <stdio.h>


/*
* This code is configured for a GTX 1060M with 1024 CUDA cores
*/


#define HEIGHT 1080
#define WIDTH 1920
#define SIZE (3 * WIDTH * HEIGHT)


__managed__ char GX[9] = { 1, 0, -1,
						   2, 0, -2,
						   1, 0, -1 };
__managed__ char GY[9] = { 1, 2, 1,
				   		   0, 0, 0,
				  		  -1,-2,-1 };


__managed__ char gaussian_kernel[25] = { 
	1, 4, 6, 4, 1,
	4, 16, 24, 16, 4,
	6, 24, 36, 24, 6,
	4, 16, 24, 16, 4,
	1, 4, 6, 4, 1
};

unsigned char *gaussian;
unsigned char *finished;


__device__ __forceinline__
int index(int x, int y) {
	if (x >= WIDTH || y >= HEIGHT || x < 0 || y < 0)
		return -1;

	return x + y * WIDTH;
}

__device__ __forceinline__
int wrap(int val, int limit) {
	if (val < 0)
		return limit + val;
	return val % limit;
}



__global__
void greyScale(unsigned char * frame, unsigned char*greyBuffer) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int pindex = 3*index(x,y);
	int sum = (frame[pindex] + frame[pindex + 1] + frame[pindex + 2]) / 3;

	greyBuffer[pindex] = sum;
	greyBuffer[pindex+1] = sum;
	greyBuffer[pindex+2] = sum;
}


__global__
void sobelOp(unsigned char * frame, unsigned char * sobel) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	for(int p = 0; p < 3; p++){
		int xDir = 0;
		int yDir = 0;

		for (int r = -1; r < 2; r++) {
			int row = y + r;
			row = wrap(row, HEIGHT);

			for (int c = -1; c < 2; c++) {
				int col = x + c;
				col = wrap(col, WIDTH);

				int pindex = 3*index(col, row);

				xDir += frame[pindex+p] * GX[(1 - c) + (1 - r) * 3];
				yDir += frame[pindex+p] * GY[(1 - c) + (1 - r) * 3];
			}
		}

		int pindex = 3*index(x, y);
		int mag = (int)sqrt((double) xDir * xDir + yDir * yDir);
		mag = min(255, mag);

		float color = frame[pindex + p] / 256.0;
		sobel[pindex + p] = color * mag;
	}

}

__global__ void gaussian_filter(const unsigned char *gaussian_input, unsigned char *gaussian_output) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= WIDTH || row >= HEIGHT)
		return;

	for(int p = 0; p < 3; p++){
		int blur = 0;

		for(int i = -2; i < 3; i++) {
			for(int j = -2; j < 3; j++) {

				const unsigned int y = wrap(row+i, HEIGHT);
				const unsigned int x = wrap(col+j, WIDTH);

				char w = gaussian_kernel[(2-j) + (2-i) * 5];
				int pindex = 3*index(x,y);
				blur += w * gaussian_input[pindex+p];
			}
		}

		blur = min(255, blur/256);
		int pindex = 3*index(col,row);
		gaussian_output[pindex+p] = blur;
	}

}


void filter(unsigned char* frame, FilterType filtertype) {
	dim3 thread(32, 32);
	dim3 block((WIDTH+31)/32, (HEIGHT+31)/32);

	cudaMemcpy(gaussian, frame, SIZE, cudaMemcpyHostToDevice);

	switch(filtertype){
		case GREY:{
			greyScale<<<block, thread>>>(gaussian, finished);
			break;
		}
		case SOBEL:{
			sobelOp<<<block, thread>>>(gaussian, finished);
			break;
		}
		case GAUSSIAN:{
			for(int i = 0; i < 4; i++){
				gaussian_filter<<<block, thread>>>(gaussian, finished);
				cudaMemcpy(gaussian, finished, SIZE, cudaMemcpyDeviceToDevice);
			}
			gaussian_filter<<<block, thread>>>(gaussian, finished);
			break;
		}
	}

	cudaMemcpy(frame, finished, SIZE, cudaMemcpyDeviceToHost);
}


#include <stdio.h>
void initCuda() {
	cudaMalloc(&gaussian, SIZE);
	cudaMalloc(&finished, SIZE);
}

void freeCuda() {
	cudaFree(gaussian);
	cudaFree(finished);
}
