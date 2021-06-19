#include <opencv2/imgcodecs.hpp>

#include "lbp/lbp.cuh"
#include "misc/image-load.hh"
#include "misc/histo_to_file.hh"
#include <iostream>

int main()
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    //size_t nb_tiles = image.cols * image.rows / 256;
    //histo_to_file(histo, nb_tiles);

    // Python

    free(histo);
}
