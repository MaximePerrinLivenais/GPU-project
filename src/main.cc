#include <iostream>
#include <opencv2/highgui/highgui_c.h>
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
    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    // TODO use cpp opencv
    /*
    size_t nb_tiles = image.cols * image.rows / 256;
    int cluster_count = 16;
    cv::Mat sample(nb_tiles, 256, CV_32S, histo);
    sample.convertTo(sample, CV_32F);
    sample = sample.reshape(256, sample.total() / 256);

    std::cout << "nb_tiles: " << nb_tiles << std::endl;
    std::cout << "sample total: " << sample.total() << std::endl;

    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(sample, cluster_count, labels,
               cv::TermCriteria(CV_TERMCRIT_ITER, 10000, 0.000001), 3,
               cv::KMEANS_PP_CENTERS, centers);

    // size_t nb_tiles = image.cols * image.rows / 256;
    // histo_to_file(histo, nb_tiles);
    // centers = centers.reshape(1, centers.total());

    std::string filepath = "../data/image1_centroids.csv";
    auto centroids_vector = load_kmean_centroids(filepath);
    std::cout << "Centroid vector: " << centroids_vector.size() << '\n';
    // float* centroids_vector = labels.data;
    std::vector<float> vec;
    centers = centers.reshape(1, centers.total());
    centers.col(0).copyTo(vec);
    // labels.isContinuous() ? labels.data : labels.clone().data;
    // uint length = labels.total() * image.channels();

    std::cout << "vector size: " << vec.size() << '\n';
    std::cout << "centers total: " << centers.total() << '\n';
    std::cout << "centers channel: " << centers.channels() << '\n';
    std::cout << "centers[0] total: " << centers.row(0).clone().total() << '\n';
    std::cout << "centers[0,0]: " << centers.at<float>(0, 0) << '\n';
    std::cout << "all total" << centers.cols * centers.channels() * centers.rows
              << '\n';
 */
    std::string filepath = "../data/centroids.csv";
    std::vector<float> centroids_vector = load_kmean_centroids(filepath);

    auto tiles_number = image.cols * image.rows / 256;
    int* nearest_neighbors =
        k_nearest_neighbors(histo, centroids_vector.data(), tiles_number);

    unsigned char* output_image =
        reconstruct_image(nearest_neighbors, image.cols, image.rows);

    cv::Mat reconstruction(image.rows / 16, image.cols / 16, CV_8UC3,
                           output_image);

    cv::imwrite("reconstruction.png", reconstruction);

    std::cout << lut[0][1] << '\n';

    free(histo);
}
