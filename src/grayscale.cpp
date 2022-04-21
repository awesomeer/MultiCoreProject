#include "grayscale.h"
#include "rgb.h"

void grayscaleInit()
{
    // Not used
}

void grayscaleFilter(uint8_t* frame, size_t rows, size_t cols)
{
    RGB* buf = (RGB*)frame;
    for (size_t i = 0; i < rows * cols; i++)
    {
        // https://en.wikipedia.org/wiki/Grayscale

        float tmp = 0.2126f * buf[i].r + 0.7152f * buf[i].g + 0.0722f * buf[i].b;
        uint8_t gray = (uint8_t)tmp;
        buf[i] = {gray, gray, gray};
    }
}

void grayscaleFree()
{
    // Not used
}
