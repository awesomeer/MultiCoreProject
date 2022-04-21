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

// __managed__ char gaussian_kernel[9] = { 
// 	1, 2, 1,
// 	2, 4, 2,
// 	1, 2, 1,
// };

__managed__ char gaussian_kernel[25] = { 
	1, 4, 6, 4, 1,
	4, 16, 24, 16, 4,
	6, 24, 36, 24, 6,
	4, 16, 24, 16, 4,
	1, 4, 6, 4, 1
};


// __managed__ char gaussian_kernel[9] = { 
// 	0, -1, 0,
// 	-1, 5, -1,
// 	0, -1, 0,
// };


unsigned char* greyScaleBuffer;
int *sobel; //1280x720
unsigned char *gaussian;
unsigned char *finished; //1280x720*3


__global__
void greyScale(unsigned char * frame, unsigned char*greyBuffer) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int index = x + y * WIDTH;
	int sum = (frame[3*index] + frame[3*index + 1] + frame[3*index + 2]) / 3;
	greyBuffer[index] = sum;
}

__global__
void greycopy(unsigned char * grey, unsigned char * frame){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int pindex = x + y * WIDTH;
	frame[3*pindex] = grey[pindex];
	frame[3*pindex+1] = grey[pindex];
	frame[3*pindex+2] = grey[pindex];
}

__device__
int index(int x, int y) {
	if (x >= WIDTH || y >= HEIGHT || x < 0 || y < 0)
		return -1;

	return x + y * WIDTH;
}

__device__ __forceinline__
int wrap(int val, int limit) {
	if (val < 0)
		return limit - 1;
	return val % limit;
}

__global__
void sobelOp(unsigned char * greyBuffer, int * sobelBuffer) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int xDir[3] = {0,0,0};
	int yDir[3] = {0,0,0};

	for (int r = -1; r < 2; r++) {
		int row = y + r;
		row = wrap(row, HEIGHT);

		for (int c = -1; c < 2; c++) {
			int col = x + c;
			col = wrap(col, WIDTH);

			int pindex = 3*index(col, row);

			for(int i = 0; i < 3; i++){
				xDir[i] += greyBuffer[pindex+i] * GX[(1 - c) + (1 - r) * 3];
				yDir[i] += greyBuffer[pindex+i] * GY[(1 - c) + (1 - r) * 3];	
			}
		}
	}

	__syncthreads();

	int pindex = 6*(index(x,y));
	for(int i = 0; i < 3; i++){
		sobelBuffer[pindex+(i*2)] = xDir[i];
		sobelBuffer[pindex+(i*2)+1] = yDir[i];
	}

}

__global__ void gaussian_filter(const unsigned char *gaussian_input, unsigned char *gaussian_output) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= WIDTH || row >= HEIGHT)
		return;

	int blur[3] = {0,0,0};
	for(int i = -2; i < 3; i++) {
		for(int j = -2; j < 3; j++) {

			const unsigned int y = max(0, min(HEIGHT - 1, row + i));
			const unsigned int x = max(0, min(WIDTH - 1, col + j));

			char w = gaussian_kernel[(2-j) + (2-i) * 5];
			//printf("%f\n", w);
			int pindex = 3*index(x,y);
			for(int i = 0; i < 3; i++){
				blur[i] += w * gaussian_input[pindex+i];
			}
		}
	}

	int pindex = 3*index(col, row);
	for(int i = 0; i < 3; i++){	
		blur[i] = min(255, blur[i]/256);
		float color = gaussian_input[pindex+i] / 256.0;
		gaussian_output[pindex+i] = (unsigned char) (((float)blur[i])*color);
	}

}

__global__
void render(int* sobolBuffer, unsigned char* frame) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int index = x + y * WIDTH;

	for(int i = 0; i < 3; i++){
		int xv = sobolBuffer[(6*index) + (2*i)];
		int yv = sobolBuffer[(6*index) + (2*i)+1];
		int mag = (int)sqrt((double) xv * xv + yv * yv);
		mag = min(255, mag);

		float color = frame[(3 * index) + i] / 256.0;
		frame[(3 * index) + i] = color * mag;	
	}
}

void filter(unsigned char* frame, FilterType filtertype) {
	dim3 thread(32, 32);
	dim3 block(WIDTH/32 + 1, HEIGHT/32 + 1);
	//dim3 block(40, 23);

	cudaMemcpy(finished, frame, SIZE, cudaMemcpyHostToDevice);

	switch(filtertype){
		case GREY:{
			greyScale<<<block, thread>>>(finished, greyScaleBuffer);
			greycopy<<<block, thread>>>(greyScaleBuffer, finished);
			break;
		}
		case SOBEL:{
			sobelOp<<<block, thread>>>(finished, sobel); //Compute Sobel convolution
			render << <block, thread >> > (sobel, finished);
			break;
		}
		case GAUSSIAN:{
			cudaMemcpy(gaussian, finished, SIZE, cudaMemcpyDeviceToDevice);
			gaussian_filter<<<block, thread>>>(gaussian, finished);
			break;
		}
	}

	cudaDeviceSynchronize();
	cudaMemcpy(frame, finished, SIZE, cudaMemcpyDeviceToHost);
}


#include <stdio.h>
void initCuda() {
	cudaMalloc(&greyScaleBuffer, WIDTH * HEIGHT);
	cudaMalloc(&sobel, sizeof(int) * WIDTH * HEIGHT * 6);
	cudaMalloc(&gaussian, sizeof(unsigned char) * WIDTH * HEIGHT * 3);
	cudaMalloc(&finished, SIZE);
}

void freeCuda() {
	cudaFree(greyScaleBuffer);
	cudaFree(sobel);
	cudaFree(gaussian);
	cudaFree(finished);
}
