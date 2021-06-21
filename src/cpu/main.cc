#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "pipeline.hh"



int main()
{
    std::string file_path = "../data/images/barcode-00-01.jpg";

    std::string image_path = cv::samples::findFile(file_path);
    auto image =  cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    auto color_image = full_pipeline(image, 16);

    cv::imwrite("output.png", color_image);

    return 0;
}
