#pragma once

#include <string>
#include <vector>

using centroid_type = double;

std::vector<centroid_type> load_kmean_centroids(const std::string& filepath);

void k_nearest_neighbors(const int* histo_tab, const float* clusters, const size_t tiles_number);

