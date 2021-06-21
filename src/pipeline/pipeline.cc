#include "pipeline/pipeline.hh"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include "lbp/lbp.cuh"
#include "misc/build_lut.hh"
#include "misc/histo_to_file.hh"
#include "misc/load_kmeans.hh"
#include "misc/reconstruct_image.cuh"
#include "neighbors/knn.cuh"

void launch_pipeline(const cv::Mat& image)
{
    // Remove warning
    (void)lut;

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    size_t nb_tiles = image.cols * image.rows / 256;
    histo_to_file(histo, nb_tiles);

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
}
