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


#define HEIGHT 720
#define WIDTH 1280
#define SIZE (3 * WIDTH * HEIGHT)


__managed__ char GX[9] = { 1, 0, -1,
						  2, 0, -2,
							1, 0, -1 };
__managed__ char GY[9] = { 1, 2, 1,
				   0, 0, 0,
				  -1,-2,-1 };

__managed__ char gaussian_kernel[9] = { 
	1/16, 1/8, 1/16,
	1/8, 1/4, 1/8,
	1/16, 1/8, 1/16,
};


unsigned char* greyScaleBuffer;
int *sobel; //1280x720
int *gaussian;
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

	int xDir = 0;
	int yDir = 0;

	for (int r = -1; r < 2; r++) {
		int row = y + r;
		row = wrap(row, HEIGHT);

		for (int c = -1; c < 2; c++) {
			int col = x + c;
			col = wrap(col, WIDTH);

			xDir += greyBuffer[index(col, row)] * GX[(1 - c) + (1 - r) * 3];
			yDir += greyBuffer[index(col, row)] * GY[(1 - c) + (1 - r) * 3];
		}
	}

	__syncthreads();

	sobelBuffer[2*index(x, y)] = xDir;
	sobelBuffer[2*index(x, y)+1] = yDir;

}

__global__ void gaussian_filter(const unsigned char *gaussian_input, int *gaussian_output) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= WIDTH || row >= HEIGHT)
		return;

    if(row < HEIGHT && col < WIDTH) {
        float blur = 0.0;
        for(int i = -1; i < 2; i++) {
            for(int j = -1; j < 2; j++) {

                const unsigned int y = max(0, min(HEIGHT - 1, row + i));
                const unsigned int x = max(0, min(WIDTH - 1, col + j));

                const float w = gaussian_kernel[(j + 1) + (i + 1) * 3];
                blur += w * gaussian_input[x + y * WIDTH];
            }
        }
        gaussian_output[col + row * WIDTH] = static_cast<unsigned char>(blur);
    }
}

__global__
void render(int* sobolBuffer, unsigned char* frame, unsigned short time) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int index = x + y * WIDTH;

	float red = frame[3 * index] / 256.0;
	float green = frame[(3 * index) + 1] / 256.0;
	float blue = frame[(3 * index) + 2] / 256.0;

	int xv = sobolBuffer[2 * index];
	int yv = sobolBuffer[2 * index + 1];
	int mag = (int)sqrt((double) xv * xv + yv * yv);
	if (mag > 255)
		mag = 255;

	frame[3 * index] = red * mag;
	frame[(3 * index) + 1] = green * mag;
	frame[(3 * index) + 2] = blue * mag;
}

void filter(unsigned char* frame) {
	static unsigned short count = 0;
	dim3 thread(32, 32);
	dim3 block(WIDTH/32 + 1, HEIGHT/32 + 1);
	//dim3 block(40, 23);

	cudaMemcpy(finished, frame, SIZE, cudaMemcpyHostToDevice);
	greyScale<<<block, thread>>>(finished, greyScaleBuffer); //Convert to grayscale

	/* sobel */
	// sobelOp<<<block, thread>>>(greyScaleBuffer, sobel); //Compute Sobel convolution
	// render << <block, thread >> > (sobel, finished, count++);

	/* gaussian */
	gaussian_filter<<<block, thread>>>(greyScaleBuffer, gaussian); //Compute Sobel convolution	
	render << <block, thread >> > (gaussian, finished, count++);

	count &= 0x3FF;

	cudaDeviceSynchronize();
	cudaMemcpy(frame, finished, SIZE, cudaMemcpyDeviceToHost);
}


#include <stdio.h>
void initCuda() {
	cudaMalloc(&greyScaleBuffer, WIDTH * HEIGHT);
	cudaMalloc(&sobel, sizeof(int) * WIDTH * HEIGHT * 2);
	cudaMalloc(&gaussian, sizeof(int) * WIDTH * HEIGHT * 2);
	cudaMalloc(&finished, SIZE);
}

void freeCuda() {
	cudaFree(greyScaleBuffer);
	cudaFree(sobel);
	cudaFree(gaussian);
	cudaFree(finished);
}