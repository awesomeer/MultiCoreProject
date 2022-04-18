#include <iostream>
#include <opencv2/opencv.hpp>

#include "sobel.h"
#include "gaussian.h"
#include "grayscale.h"

enum class FilterType : size_t{
    SOBEL = 0,
    GAUSSIAN = 1,
    GRAYSCALE = 2
};

int main(int argc, char** argv) {

    cv::VideoCapture camera;

    if (argc > 1)
        camera.open(argv[1], cv::CAP_ANY);
    else
        camera.open(0, cv::CAP_ANY);

    if (!camera.isOpened())
        return -1;
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    cv::Mat cap;

    cv::namedWindow("Video Stream");
    cv::resizeWindow("Video Stream", 1920, 1080);

    sobelInit();
    gaussianInit();
    grayscaleInit();

    FilterType active_filter = FilterType::SOBEL;
    bool should_stop = false;

    while (!should_stop) {
        camera >> cap;
        if (cap.empty())
            break;

        switch (active_filter)
        {
        case FilterType::SOBEL:
            sobelFilter(cap.data, cap.rows, cap.cols);
            break;
        case FilterType::GAUSSIAN:
            gaussianFilter(cap.data, cap.rows, cap.cols);
            break;
        case FilterType::GRAYSCALE:
            grayscaleFilter(cap.data, cap.rows, cap.cols);
            break;
        default:
            break;
        }

        cv::imshow("Video Stream", cap);

        switch (cv::waitKey(10)) {
        case 27: // ESC
            should_stop = true;
            break;
        case 49: // 1
            active_filter = FilterType::SOBEL;
            break;
        case 50: // 2
            active_filter = FilterType::GAUSSIAN;
            break;
        case 51: // 3
            active_filter = FilterType::GRAYSCALE;
            break;
        default:
            break;
            }
    }
    std::cout << cap.size << std::endl;

    sobelFree();
    gaussianFree();
    grayscaleFree();

    return 0;
}