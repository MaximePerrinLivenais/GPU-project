#pragma once

#include <string>
#include <vector>

using centroid_type = double;

std::vector<centroid_type> load_kmean_centroids(const std::string& filepath);
