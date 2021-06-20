#include "build_lut.hh"
#include "reconstruct_image.cuh"

__global__ void reconstruct_image_kernel(unsigned char* reconstruction,
                                         // const size_t reconstruction_pitch,
                                         const int* nearest_neighbors,
                                         const size_t nearest_neighbors_size,
                                         const unsigned char* lut,
                                         const size_t lut_size)
{
    // int x = blockDim.x * blockIdx.x + threadIdx.x;
    // int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tile_number = blockIdx.y * gridDim.x + blockIdx.x;
    // int tile_number = y * reconstruction_pitch + x;

    if (tile_number >= nearest_neighbors_size)
        return;

    auto centroid = nearest_neighbors[tile_number];
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

    // Reconstructed image
    unsigned char* reconstruction;
    // size_t reconstruction_pitch = 0;

    rc = cudaMalloc(&reconstruction,
                    image_cols / 16 * 3 * sizeof(unsigned char) * image_rows
                        / 16);
    // rc = cudaMallocPitch(&reconstruction, &reconstruction_pitch,
    //                     image_cols / 16 * 3 * sizeof(unsigned char),
    //                     image_rows / 16);

    if (rc)
    {
        std::cout << "Could not allocate memory for image reconstruction\n";
        exit(EXIT_FAILURE);
    }

    // Nearest neighbors
    int* cuda_nearest_neighbor;
    size_t pixels_number = image_rows * image_cols;
    size_t cuda_nearest_neighbors_size = pixels_number / 256;

    rc = cudaMalloc(&cuda_nearest_neighbor, pixels_number / 256 * sizeof(int));
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }
    rc = cudaMemcpy(cuda_nearest_neighbor, nearest_neighbors,
                    pixels_number / 256 * sizeof(int), cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout
            << "Could not copy nearest neighbors result from host to device\n";
        exit(EXIT_FAILURE);
    }

    // Look up table
    unsigned char* cuda_lut;
    size_t cuda_lut_size = 16 * 3;

    rc = cudaMalloc(&cuda_lut, 16 * 3 * sizeof(unsigned char));
    if (rc)
    {
        std::cout << "Could not allocate memory for the nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy(cuda_lut, lut, 16 * 3 * sizeof(unsigned char),
                    cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy lut from host to device\n";
        exit(EXIT_FAILURE);
    }

    int bsize = 16;
    int tile_in_width = std::ceil((float)image_cols / bsize);
    int tile_in_height = std::ceil((float)image_rows / bsize);

    dim3 reconstruction_dim_grid(tile_in_width, tile_in_height);

    // int block_size = 32;
    // int w = std::ceil((float)tile_in_width / block_size);
    // int h = std::ceil((float)tile_in_height / block_size);

    // dim3 reconstruction_dim_grid(w, h);
    // dim3 reconstruction_dim_block(block_size, block_size);

    reconstruct_image_kernel<<<reconstruction_dim_grid,
                               1 /*reconstruction_dim_block*/>>>(
        reconstruction, /*reconstruction_pitch,*/ cuda_nearest_neighbor,
        cuda_nearest_neighbors_size, cuda_lut, cuda_lut_size);

    cudaDeviceSynchronize();

    unsigned char* output = (unsigned char*)malloc(
        tile_in_width * tile_in_height * 3 * sizeof(unsigned char));

    // cudaMemcpy2D(output, tile_in_width * 3 * sizeof(unsigned char),
    //             reconstruction, reconstruction_pitch,
    //             tile_in_width * 3 * sizeof(unsigned char), tile_in_height,
    //             cudaMemcpyDeviceToHost);
    cudaMemcpy(output, reconstruction, tile_in_height * tile_in_width * 3,
               cudaMemcpyDeviceToHost);

    cudaFree(reconstruction);
    cudaFree(cuda_lut);
    cudaFree(cuda_nearest_neighbor);

    return output;
}
