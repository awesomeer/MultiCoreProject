#pragma once


enum FilterType {GREY, SOBEL, GAUSSIAN};

void initCuda();

void filter(unsigned char* frame, FilterType filtertype);

void freeCuda();