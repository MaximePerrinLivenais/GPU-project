#pragma once

#include <vector>
#include <opencv2/imgcodecs.hpp>


std::vector<std::vector<int>> compute_lbp_values_cpu(const cv::Mat& image, size_t tile_size);
