#include <iostream>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

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
    //size_t blur_size = 3;
    //float* blur_mat = (float*)malloc(blur_size * blur_size * sizeof(float));
    //std::fill_n(blur_mat, blur_size * blur_size, 1.0f / (float)(blur_size * blur_size));

    while (!should_stop) {
        camera >> cap;
        if (cap.empty())
            break;

        auto start = std::chrono::high_resolution_clock::now();

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

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> frame_time = end - start;
        float fps = 1.0f / frame_time.count();
        cv::String fps_text = std::to_string(fps).substr(0, 5) +" FPS";
        cv::Point org = cv::Point(cap.cols - 200, 30);
        int font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        cv::Scalar color = { 0, 255, 0 };
        cv::putText(cap, fps_text, org, font, font_scale, color);
        cv::imshow("Video Stream", cap);

        switch (cv::waitKey(10)) {
        case 27: // ESC
            should_stop = true;
            break;
        case 49: // 1
            active_filter = FilterType::SOBEL;
            std::cout << "Sobel" << std::endl;
            break;
        case 50: // 2
            active_filter = FilterType::GAUSSIAN;
            std::cout << "Gaussian Blur" << std::endl;
            break;
        case 51: // 3
            active_filter = FilterType::GRAYSCALE;
            std::cout << "Grayscale" << std::endl;
            break;
        //case 46: //.
        //    blur_size++;
        //    free(blur_mat);
        //    blur_mat = (float*)malloc(blur_size * blur_size * sizeof(float));
        //    std::fill_n(blur_mat, blur_size * blur_size, 1.0f / (float)(blur_size * blur_size));
        //    std::cout << "Blur size " << blur_size << std::endl;
        //    break;
        //case 44: //,
        //    if (blur_size > 1) {
        //        blur_size--;
        //        free(blur_mat);
        //        blur_mat = (float*)malloc(blur_size * blur_size * sizeof(float));
        //        std::fill_n(blur_mat, blur_size * blur_size, 1.0f / (float)(blur_size * blur_size));
        //    }
        //    std::cout << "Blur size " << blur_size << std::endl;
        //    break;
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