#include "histo_to_file.hh"
#include <fstream>
#include <stddef.h>

void histo_to_file(int *histo, size_t nb_tiles)
{
    std::ofstream file("histo.csv");

    size_t tile_size = 256;
    for (size_t i = 0; i < nb_tiles; i++)
    {
        for (size_t j = 0; j < tile_size - 1; j++)
        {
            file << histo[i * tile_size + j] << ';';
        }

        file << histo[i * tile_size + 255] << "\n";
    }
}
