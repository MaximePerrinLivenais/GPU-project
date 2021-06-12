#include "image-load.hh"

#include <opencv2/imgcodecs.hpp>

cv::Mat load_image(const char* file_path)
{
    std::string image_path = cv::samples::findFile(file_path);
    return cv::imread(image_path, cv::IMREAD_GRAYSCALE);
}
