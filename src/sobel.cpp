#include "sobel.h"
#include "rgb.h"


void sobelInit() {
    // Dummy function. Only runs when cuda is not available.
}

void sobelFilter(uint8_t* frame, size_t rows, size_t cols){
    // Not a sobel filter
    RGB* buf = (RGB*)frame;
    for (size_t i = 0; i < rows*cols; i++)
    {
        RGB tmp = { buf[i].g, buf[i].b, buf[i].r };
        buf[i] = tmp;
    }
}

void sobelFree() {
    // Dummy function. Only runs when cuda is not available.
}