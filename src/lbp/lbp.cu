#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "lbp.cuh"

__global__ void lbp_value_kernel(const unsigned char* image,
                                 unsigned char* lbp_values, const int width,
                                 const int height, const size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    auto pixel_value = image[y * width + x];

    unsigned char lbp_value = 0;

    lbp_value |= (threadIdx.y > 0
                    && image[(y - 1) * width + x] >= pixel_value) << 7;
    lbp_value |= (threadIdx.y > 0 && threadIdx.x > 0
                    && image[(y - 1) * width + x - 1] >= pixel_value) << 6;
    lbp_value |= (threadIdx.x > 0
                    && image[y * width + x - 1] >= pixel_value) << 5;
    lbp_value |= (threadIdx.y < 15 && threadIdx.x > 0
                    && image[(y + 1) * width + x - 1] >= pixel_value) << 4;
    lbp_value |= (threadIdx.y < 15
                    && image[(y + 1) * width + x] >= pixel_value) << 3;
    lbp_value |= (threadIdx.y < 15 && threadIdx.x < 15
                    && image[(y + 1) * width + x + 1] >= pixel_value) << 2;
    lbp_value |= (threadIdx.x < 15
                    && image[y * width + x + 1] >= pixel_value) << 1;
    lbp_value |= threadIdx.y > 0 && threadIdx.x < 15
                    && image[(y - 1) * width + x + 1] >= pixel_value;

    auto tile_index = blockIdx.y * width / 16 + blockIdx.x;
    auto pixel_index_in_tile = blockDim.x * threadIdx.y + threadIdx.x;

    auto lbp_index = lbp_values + tile_index * pitch
        + pixel_index_in_tile;
    *lbp_index = lbp_value;

    // printf("(%d, %d) = %u\n", x, y, lbp_value);
}

__global__ void compute_histo_kernel(int* histo_tab,
                                        const size_t histo_pitch,
                                        unsigned char* lbp_values,
                                        const size_t lbp_pitch,
                                        const size_t height)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    if (x >= 256 || y >= height)
        return;

    unsigned char* lbp_index = lbp_values + y * lbp_pitch + x;
    int* histo_index =  histo_tab + y * histo_pitch + *lbp_index;

    //assert(y == 0);
    //printf("histo_tab: %p, histo_index: %p, lbp_value: %x\n", histo_tab, histo_index, *lbp_index * sizeof(int));
    //printf("LBP_INDEX: %d - y * 256 + x: %d\nhisto_index: %d\n\n", *lbp_index, y * 256 + x, histo_index);

    atomicAdd(histo_index, 1);
    __syncthreads();

    //printf("histo_value: %d at %d\n", *histo_index, *lbp_index);
}

void compute_lbp_values(const unsigned char* image, const size_t width,
                        const size_t height)
{
    cudaError_t rc = cudaSuccess;

    unsigned char* cuda_image;
    auto pixels_number = width * height;

    rc = cudaMalloc(&cuda_image, pixels_number * sizeof(unsigned char));
    if (rc)
    {
        std::cout << "Could not allocate memory for the image on the device\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy(cuda_image, image, pixels_number * sizeof(unsigned char),
                    cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy image data from host to device\n";
        exit(EXIT_FAILURE);
    }

    unsigned char* lbp_values;
    size_t lbp_pitch;
    auto tiles_number = width * height / 256;

    rc = cudaMallocPitch(&lbp_values, &lbp_pitch, 256 * sizeof(unsigned char), tiles_number);
    if (rc)
    {
        std::cout << "Could not allocate memory for lbp values buffer\n";
        exit(EXIT_FAILURE);
    }

    int bsize = 16;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    /*std::cout << "Running kernel of size ("
              << w << ", " << h << ")\n";*/

    dim3 lbp_dim_block(bsize, bsize);
    dim3 lbp_dim_grid(w, h);

    lbp_value_kernel<<<lbp_dim_grid, lbp_dim_block>>>(cuda_image, lbp_values, width, height, lbp_pitch);

    cudaDeviceSynchronize();

    int* histo_tab;
    size_t histo_pitch;
    rc = cudaMallocPitch(&histo_tab, &histo_pitch, 256 * sizeof(int), tiles_number);
    if (rc)
    {
        std::cout << "Could not allocate memory for lbp values buffer\n";
        exit(EXIT_FAILURE);
    }

    cudaMemset2D(histo_tab, histo_pitch, 0, 256 * sizeof(int), tiles_number);

    int histo_threads = 256;
    int histo_blocks = tiles_number;

    std::cout << "lbp_pitch: " << lbp_pitch << ", " << "histo_pitch: " << histo_pitch << "\n";

    compute_histo_kernel<<<1, histo_threads>>>(histo_tab, histo_pitch, lbp_values, lbp_pitch, tiles_number);

    cudaDeviceSynchronize();
    /*std::cout << (int) image[0] << " " << (int) image[1] << " " << (int)
    image[2] << "\n"
                << (int) image[width] << " " << (int) image[width + 1] << "
    " << (int) image[width + 2] << "\n"
                << (int) image[2 * width] << " " << (int) image[2 * width +
    1]
    << " " << (int) image[2 * width + 2] << "\n";
    */

    int* output = (int*) malloc(256 * tiles_number * sizeof(int));
    if (output == NULL)
    {
        std::cout << "CRINGE\n";
        exit(EXIT_FAILURE);
    }

    cudaMemcpy2D(output, 256 * sizeof(int), histo_tab, histo_pitch, 256 * sizeof(int), tiles_number, cudaMemcpyDeviceToHost);

    std::cout << "HISTO\n" << sizeof(unsigned char) << "\n";

    auto sum = 0;
    for (auto i = 0; i < 256 * tiles_number; i++)
    {
       int value = *(output + i);
       std::cout << i << ": " << value  << "\n";
       sum += value;
    }

    assert(sum == 256 * tiles_number);

    std::cout << "Sum: " << sum << "\n";

    std::free(output);

    cudaFree(cuda_image);
    cudaFree(lbp_values);
    cudaFree(histo_tab);
}
