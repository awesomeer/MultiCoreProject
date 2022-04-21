#include "gaussian.h"
#include <vector>
#include "convolution.hpp"

float gaussian[9] = {1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f,
                     1.0f / 8.0f,  1.0f / 4.0f, 1.0f / 8.0f,
                     1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f };

void gaussianInit() {
    // Not used
}

void gaussianFilter(uint8_t* frame, size_t rows, size_t cols) {
    std::vector<RGB> buf((RGB*)frame, (RGB*)frame + rows * cols);
    convolve(buf.data(), (RGB*)frame, { cols, rows }, gaussian, { 3, 3 }, { -1, -1 });
}

void gaussianFree() {
    // Not used
}