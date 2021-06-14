#include "lbp.cuh"

#include <cassert>
#include <stdio.h>
#include <iostream>

__global__ void lbp_value_kernel(const unsigned char* image,
                                    unsigned char* lbp_values,
                                    const int width,
                                    const int height,
                                    const size_t pitch)
{
    /*
    #ifdef __CUDA_ARCH__
        printf("CUDA architecture of the current running device code: %d.\n", __CUDA_ARCH__);
        printf("Image height: %d. Image width: %d.\n", height, width);
    #endif
    */

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    auto tile_index = blockIdx.y * width / 16 + blockIdx.x;
    if (tile_index != 510 || x >= width || y >= height)
        return;

    //printf("%d\n", tile_index);
    printf("(%d, %d)\n", x, y);
}

void compute_lbp_values(const unsigned char *image, const size_t width, const size_t height)
{
    cudaError_t rc = cudaSuccess;

    auto tiles_number = width * height / 256;

    //std::cout << "Tiles : " << tiles_number << "\n";

    unsigned char* lbp_values;
    size_t pitch;

    rc = cudaMallocPitch(&lbp_values, &pitch, 256 * sizeof(unsigned char), tiles_number);
    if (rc)
    {
        std::cout << "Error in cudaMallocPitch\n";
        exit(1);
    }

    int bsize = 16;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    std::cout << "Running kernel of size ("
              << w << ", " << h << ")\n";

    dim3 dim_block(bsize, bsize);
    dim3 dim_grid(w, h);

    lbp_value_kernel<<<dim_grid, dim_block>>>(image, lbp_values, width, height, pitch);

    cudaDeviceSynchronize();
}
