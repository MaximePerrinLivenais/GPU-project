#include <cmath>
#include <string>
#include "knn.hh"
#include "../misc/load_kmeans.hh"


double euclidian_distance(const std::vector<float>& centroid,
        const std::vector<int>& histogram)
{
    double euclidian_distance = 0;

    for (size_t i = 0; i < centroid.size(); i++)
        euclidian_distance += (centroid[i] - histogram[i]) * (centroid[i] - histogram[i]);

    return std::sqrt(euclidian_distance);
}

int compute_nearest_neighbors(const std::vector<std::vector<float>>& centroids,
        const std::vector<int>& histogram)
{

    std::vector<double> distances(centroids.size(), 0);

    for (size_t i = 0; i < centroids.size(); i++)
    {
        auto dist = euclidian_distance(centroids[i], histogram);
        distances[i] = dist;
    }

    int nearest_index = 0;

    for (size_t i = 1; i < distances.size(); i++)
    {
        if (distances[nearest_index] > distances[i])
            nearest_index = i;
    }

    return nearest_index;
}


std::vector<int> knn(const std::vector<std::vector<int>>& histograms,
        const std::vector<std::vector<float>>& centroids)
{


    std::vector<int> nearest_neighbors;
    for (auto histogram : histograms)
    {
        auto index = compute_nearest_neighbors(centroids, histogram);
        nearest_neighbors.push_back(index);
    }

    return nearest_neighbors;
}

std::vector<int> compute_knn(const std::string& filepath,
        const std::vector<std::vector<int>>& histograms,
        size_t tile_size)
{
    auto centroids_vect = load_kmean_centroids(filepath);

    std::vector<std::vector<float>> centroids_reshape;

    size_t i = 0;
    std::vector<float> current_centroids;
    for (auto centroid : centroids_vect)
    {
        current_centroids.push_back(centroid);
        i++;

        if (i >= tile_size * tile_size)
        {
            centroids_reshape.push_back(current_centroids);
            current_centroids = std::vector<float>();
            i = 0;
        }
    }

    return knn(histograms, centroids_reshape);
}
