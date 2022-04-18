#pragma once

#include <cstdint>
#include <cstddef>

void grayscaleInit();
void grayscaleFilter(uint8_t* frame, size_t rows, size_t cols);
void grayscaleFree();