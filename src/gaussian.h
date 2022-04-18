#pragma once

#include <cstdint>
#include <cstddef>

void gaussianInit();
void gaussianFilter(uint8_t* frame, size_t rows, size_t cols);
void gaussianFree();