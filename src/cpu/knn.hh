#pragma once

#include <vector>

std::vector<int> compute_knn(const std::string& filepath,
        const std::vector<std::vector<int>>& histograms,
        size_t tile_size);
