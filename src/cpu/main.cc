#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "pipeline.hh"


int main(int argc, char *argv[])
{
    std::string image_path = cv::samples::findFile(argv[1]);
    auto image =  cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    auto color_image = full_pipeline(image, 16);

    cv::imwrite("output.png", color_image);

    return 0;
}
