#include <opencv2/imgcodecs.hpp>

#include "lbp/lbp.cuh"
#include "neighbors/knn.cuh"
#include "misc/image-load.hh"
#include "misc/histo_to_file.hh"
#include "misc/load_kmeans.hh"

#include <iostream>
#include <unistd.h>
#include <sys/wait.h>


void launch_python_kmeans(const std::string script_name,
        const std::string output_file)
{
    pid_t pid = fork();
    if (pid == -1)
    {
        std::cout << "Error fork\n";
        exit(EXIT_FAILURE);
    }
    else if (pid > 0) // Parent
    {
        std::cout << "Waiting child: " << pid << ".\n";

        int return_status;
        waitpid(pid, &return_status, 0);

        std::cout << "Child returns " << return_status << ".\n";
    }
    else
    {
        char *argv_list[] = { (char *)"python", (char *)script_name.c_str(),
            (char *)output_file.c_str(), (char*)NULL };

        execvp("python", argv_list);
        exit(EXIT_FAILURE);
    }
}


int main()
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    int* histo = compute_lbp_values(image.data, image.cols, image.rows);

    //size_t nb_tiles = image.cols * image.rows / 256;
    //histo_to_file(histo, nb_tiles);

    std::string filepath = "test.csv";
    auto centroids_vector = load_kmean_centroids(filepath);

    auto tiles_number = image.cols * image.rows / 256;
    k_nearest_neighbors(histo, centroids_vector.data(), tiles_number);

    free(histo);
}
