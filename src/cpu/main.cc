#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "lbp.hh"

int main()
{
    std::string file_path = "../data/images/barcode-00-01.jpg";
    std::string image_path = cv::samples::findFile(file_path);
    auto image =  cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    int x = (int)(image.at<uchar>(0,0));
    std::cout << x << "\n";

    auto histo = compute_lbp_values(image, 16);
    std::cout << histo.size() << "\n";
    histo_to_file(histo);
}
