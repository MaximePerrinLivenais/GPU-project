#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include "lbp/lbp.cuh"
#include "misc/build_lut.hh"
#include "misc/histo_to_file.hh"
#include "misc/image-load.hh"
#include "misc/load_kmeans.hh"
#include "misc/reconstruct_image.cuh"
#include "neighbors/knn.cuh"

int main()
{
    /* To deal with unused error because only used in cuda code */
    (void) lut;

    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    std::string filepath = "../data/image1_centroids.csv";
    auto centroids_vector = load_kmean_centroids(filepath);

    auto tiles_number = image.cols * image.rows / 256;
    int* nearest_neighbors =
        k_nearest_neighbors(histo, centroids_vector.data(), tiles_number);

    unsigned char* output_image =
        reconstruct_image(nearest_neighbors, image.cols, image.rows);

    cv::Mat reconstruction(image.rows / 16, image.cols / 16, CV_8UC3,
                           output_image);

    cv::imwrite("reconstruction.png", reconstruction);

    free(histo);
    free(nearest_neighbors);
    free(output_image);
}
