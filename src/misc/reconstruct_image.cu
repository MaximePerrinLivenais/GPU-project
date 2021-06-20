#include "build_lut.hh"

int* reconstruct_image(int* nearest_neighbors, int image_cols, int image_rows)
{
    cudaError_t rc = cudaSuccess;

    // Reconstructed image
    unsigned char* reconstruction;
    size_t reconstruction_pitch;

    rc = cudaMallocPitch(&reconstruction, &reconstruction_pitch, image_cols * sizeof(unsigned char), image_rows);

    if (rc)
    {
        std::cout << "Could not allocate memory for image reconstruction\n";
        exit(EXIT_FAILURE);
    }

    // Nearest neighbors
    int* cuda_nearest_neighbor;
    size_t pixel_number = image_rows * image_cols;

    rc = cudaMalloc(&cuda_nearest_neighbor, pixels_number / 256 * sizeof(unsigned char));
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    // Look up table
    unsigned char* cuda_lut;

    rc = cudaMalloc(&cuda_lut, 16 * 3 * sizeof(unsigned char);
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }
}
