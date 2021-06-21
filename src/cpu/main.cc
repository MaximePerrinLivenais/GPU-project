#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "pipeline.hh"



int main()
{
    std::string file_path = "../data/images/barcode-00-01.jpg";

    auto color_image = full_pipeline(file_path, 16);

    cv::imwrite("output.png", color_image);

    return 0;
}
