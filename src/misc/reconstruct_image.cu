#include "reconstruct_image.cuh"

#include <cassert>

#include "build_lut.hh"

__global__ void reconstruct_image_kernel(unsigned char* reconstruction,
                                         const int* nearest_neighbors,
                                         const size_t nearest_neighbors_size,
                                         const unsigned char* lut,
                                         const size_t lut_size,
                                         const size_t width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_number = y * width + x;

    if (tile_number >= nearest_neighbors_size)
        return;

    auto centroid = nearest_neighbors[tile_number];
    if (centroid >= lut_size)
        return;

    unsigned char red = lut[centroid * 3 + 0];
    unsigned char green = lut[centroid * 3 + 1];
    unsigned char blue = lut[centroid * 3 + 2];

    reconstruction[tile_number * 3 + 0] = blue;
    reconstruction[tile_number * 3 + 1] = green;
    reconstruction[tile_number * 3 + 2] = red;
}

unsigned char* reconstruct_image(int* nearest_neighbors, int image_cols,
                                 int image_rows)
{
    cudaError_t rc = cudaSuccess;

    size_t tile_size = 16;
    size_t tiles_in_width = std::ceil((float)image_cols / tile_size);
    size_t tiles_in_height = std::ceil((float)image_rows / tile_size);
    size_t tiles_number = tiles_in_height * tiles_in_width;

    // Reconstructed image
    unsigned char* reconstruction;

    rc = cudaMalloc(&reconstruction, tiles_number * 3 * sizeof(unsigned char));
    if (rc)
    {
        std::cout << "Could not allocate memory for image reconstruction\n";
        exit(EXIT_FAILURE);
    }

    // Nearest neighbors
    int* cuda_nearest_neighbors;

    rc = cudaMalloc(&cuda_nearest_neighbors, tiles_number * sizeof(int));
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy(cuda_nearest_neighbors, nearest_neighbors,
                        tiles_number * sizeof(int), cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy nearest neighbors result from host to device\n";
        exit(EXIT_FAILURE);
    }

    // Look up table
    unsigned char* cuda_lut;
    size_t lut_size = 16 * 3;

    rc = cudaMalloc(&cuda_lut, lut_size * sizeof(unsigned char));
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy(cuda_lut, lut, lut_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy lut from host to device\n";
        exit(EXIT_FAILURE);
    }

    size_t block_size = 32;
    dim3 block_dim(block_size, block_size);

    int grid_width = std::ceil((float) tiles_in_width / block_size);
    int grid_height = std::ceil((float) tiles_in_height / block_size);
    dim3 grid_dim(grid_width, grid_height);

    reconstruct_image_kernel<<<grid_dim, block_dim>>>(reconstruction, cuda_nearest_neighbors,
        tiles_number, cuda_lut, lut_size, tiles_in_width);

    cudaDeviceSynchronize();

    unsigned char* output = (unsigned char*) malloc(tiles_number * 3 * sizeof(unsigned char));

    cudaMemcpy(output, reconstruction, tiles_number * 3, cudaMemcpyDeviceToHost);

    cudaFree(reconstruction);
    cudaFree(cuda_lut);
    cudaFree(cuda_nearest_neighbors);

    return output;
}
