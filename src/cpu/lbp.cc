#include <opencv2/imgcodecs.hpp>
#include <fstream>

int compute_patch_lbp(const cv::Mat& image, size_t i, size_t j)
{
    static std::vector<std::tuple<int, int>> index = {
        {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}};


    int power = 0;
    int threshold = (int)(image.at<uchar>(i, j));
    int lbp = 0;

    for (const auto& tuple : index)
    {
        int x_index = i + std::get<0>(tuple);
        int y_index = j + std::get<1>(tuple);

        if (x_index >= 0  && x_index < image.rows && y_index >= 0 && y_index < image.cols)
        {
            int val = (int)(image.at<uchar>(x_index, y_index));
            lbp |= (val >= threshold) << power;
        }
        power++;
    }

    return lbp;
}


std::vector<int> compute_texton_histo(const cv::Mat& image)
{
    std::vector<int> texton_histo(256, 0);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
            texton_histo[compute_patch_lbp(image ,i, j)] += 1;
    }
    return texton_histo;
}

std::vector<std::vector<int>> compute_lbp_values(const cv::Mat& image,
        size_t tile_size)
{
    std::vector<std::vector<int>> image_histo;

    auto rows = image.rows / tile_size;
    auto cols = image.cols / tile_size;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            // check oob with non 16 aligned image
            auto tile = image(cv::Range(i * tile_size, (i +1) * tile_size),
                    cv::Range(j * tile_size, (j + 1) * tile_size));
            auto texton_histo = compute_texton_histo(tile);
            image_histo.push_back(texton_histo);
        }

    }

    return image_histo;
}

void histo_to_file(std::vector<std::vector<int>> histo)
{
    std::ofstream file("histo.csv");

    for (const auto& vect : histo)
    {
        for (const auto& val : vect)
            file << val << ";";
        file << "\n";
    }
}
