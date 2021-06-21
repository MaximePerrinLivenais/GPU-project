#pragma once

#include <opencv2/imgcodecs.hpp>

cv::Mat full_pipeline(const cv::Mat& image, size_t tile_size);
