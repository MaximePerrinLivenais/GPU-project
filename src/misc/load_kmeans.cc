#include "load_kmeans.hh"

#include <iostream>
#include <fstream>

void parse_line(std::string line,
        std::vector<centroid_type>& centroids,
        const std::string delimiter = ",")
{
    size_t pos = 0;
    std::string token;

    while ((pos = line.find(delimiter)) != std::string::npos)
    {
        token = line.substr(0, pos);
        centroids.emplace_back(std::stod(token));

        line.erase(0, pos + delimiter.length());
    }

    centroids.emplace_back(std::stod(line));
}

std::vector<centroid_type> load_kmean_centroids(const std::string& filepath)
{
    std::ifstream file(filepath);

    if (!file.is_open())
    {
        std::cout << "Error while opening centroids in: " << filepath << ".\n";
        exit(1);
    }

    std::string buffer;
    std::vector<centroid_type> centroids;

    while (getline(file, buffer))
    {
        if (buffer.empty())
            continue;

        parse_line(buffer, centroids);
    }

    return centroids;
}
