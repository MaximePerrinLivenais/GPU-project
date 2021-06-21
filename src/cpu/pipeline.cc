#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "lbp.hh"
#include "knn.hh"
#include <opencv2/highgui.hpp>

static cv::Mat reconstruct_image(std::vector<int> nn, size_t rows, size_t cols,
        size_t tile_size)
{
    std::cout << nn.size() << "\n";
    static unsigned char lut[16][3] = {
        { 84,   0, 255},
        {255,   0,  23},
        {  0, 116, 255},
        {255, 100,   0},
        {184,   0, 255},
        {255, 200,   0},
        {255,   0, 124},
        {  0,  15, 255},
        {255,   0,   0},
        {108, 255,   0},
        {  0, 255, 192},
        {  0, 255,  92},
        {255,   0, 224},
        {  7, 255,   0},
        {208, 255,   0},
        {  0, 216, 255}
    };

    uchar *pixels = new uchar[nn.size() * 3 * sizeof(uchar)];

    for (size_t i = 0; i < nn.size(); i++)
    {
        pixels[i * 3 + 0] = lut[nn[i]][2];
        pixels[i * 3 + 1] = lut[nn[i]][1];
        pixels[i * 3 + 2] = lut[nn[i]][0];
    }

    cv::Mat reconstructed_image(rows / tile_size, cols / tile_size, CV_8UC3,
                           pixels);

    return reconstructed_image;
}

cv::Mat full_pipeline(const std::string& file_path, size_t tile_size)
{
    std::string image_path = cv::samples::findFile(file_path);
    auto image =  cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    auto histo = compute_lbp_values_cpu(image, tile_size);
    auto knn = compute_knn("../centroids.csv", histo, tile_size);

    auto color_image = reconstruct_image(knn, image.rows, image.cols, tile_size);
    return color_image;
}

