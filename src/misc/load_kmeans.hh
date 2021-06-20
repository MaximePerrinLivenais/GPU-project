#pragma once

#include <string>
#include <vector>

using centroid_type = float;

std::vector<centroid_type> load_kmean_centroids(const std::string& filepath);
