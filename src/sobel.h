#pragma once

#include <cstdint>
#include <cstddef>

void sobelInit();
void sobelFilter(uint8_t* frame, size_t rows, size_t cols);
void sobelFree();