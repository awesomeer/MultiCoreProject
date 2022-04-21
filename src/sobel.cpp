#include "sobel.h"
#include <vector>
#include <cmath>
#include "rgb.h"
#include "convolution.hpp"

// https://de.wikipedia.org/wiki/Sobel-Operator
// Using scharr operator for better rotational symmetry
int scharr_x[9] = { 47, 0, -47,
                   162, 0, -162,
                    47, 0, -47 };
int scharr_y[9] = { 47,  162, 47,
                     0,    0,  0,
                   -47, -162,-47 };

void sobelInit() {
    // Not used
}

void sobelFilter(uint8_t* frame, size_t rows, size_t cols) {
    std::vector<Vec3D<int>> frame_x(rows*cols);
    std::vector<Vec3D<int>> frame_y(rows*cols);
    convolve((RGB*)frame, frame_x.data(), { cols, rows }, scharr_x, { 3,3 }, { -1, -1 });
    convolve((RGB*)frame, frame_y.data(), { cols, rows }, scharr_y, { 3,3 }, { -1, -1 });
    for (size_t y = 0; y < rows; y++)
    {
        for (size_t x = 0; x < cols; x++)
        {
            size_t idx = y * cols + x;
            for (size_t rgb = 0; rgb < 3; rgb++) {
                int G_x = frame_x[idx].data[rgb] / 512;
                int G_y = frame_y[idx].data[rgb] / 512;
                int res = (int)std::sqrt((double)(G_x * G_x + G_y * G_y));
                ((RGB*)frame)[idx].data[rgb] = (uint8_t)res;
            }
        }
    }
}

void sobelFree() {
    // Not used
}