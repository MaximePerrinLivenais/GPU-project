#include <opencv2/imgcodecs.hpp>

#include "lbp/lbp.cuh"
#include "neighbors/knn.cuh"
#include "misc/image-load.hh"
#include "misc/histo_to_file.hh"
#include "misc/load_kmeans.hh"
#include "misc/build_lut.hh"

#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    size_t nb_tiles = image.cols * image.rows / 256;
    histo_to_file(histo, nb_tiles);

    std::string filepath = "../data/image1_centroids.csv";
    auto centroids_vector = load_kmean_centroids(filepath);

    auto tiles_number = image.cols * image.rows / 256;
    int* nearest_neighbors = k_nearest_neighbors(histo, centroids_vector.data(), tiles_number);

    auto output_image = reconstruct_image(nearest_neighbors, image.cols, image.rows);

    std::cout << lut[0][1] << '\n';

    free(histo);
}
