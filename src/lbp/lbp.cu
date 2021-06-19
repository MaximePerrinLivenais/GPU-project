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

    // printf("Y is: %d\n", y);
    if (x >= width || y >= height)
        return;

    //printf("block y is: %d\n", gridDim.x * blockIdx.y + blockIdx.x);

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

    //printf("(%d, %d) = %u\n", x, y, lbp_value);
}

__global__ void test()
{
    printf("%d\n", gridDim.x * blockIdx.y + blockIdx.x);
}

__global__ void compute_histo_kernel(int* histo_tab,
                                        const size_t histo_pitch,
                                        unsigned char* lbp_values,
                                        const size_t lbp_pitch,
                                        const size_t height) //tile_number
{
    int lbp_index = threadIdx.y * blockDim.x + threadIdx.x;
    //int y = threadIdx.y * blockDim.y + threadIdx.x;
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;

    //printf("%d  %d  %d %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.z);

    if (lbp_index >= 256 || tile_index >= height)
        return;

    //printf("%d\n", tile_index);

    //printf("%d\n", gridDim.x * blockIdx.y + blockIdx.x);

    unsigned char lbp_value = lbp_values[tile_index * lbp_pitch + lbp_index];

    //printf("lbp: %d\n", lbp_value);
    int* histo_index =  histo_tab + tile_index * histo_pitch / sizeof(int) + lbp_value;

    //assert(y == 0);
    //printf("histo_tab: %p, histo_index: %p, lbp_value: %x\n", histo_tab, histo_index, *lbp_index * sizeof(int));
    //printf("LBP_INDEX: %d - y * 256 + x: %d\nhisto_index: %d\n\n", *lbp_index, y * 256 + x, histo_index);

    atomicAdd(histo_index, 1);
    __syncthreads();

    //printf("histo_value: %p at %d\n", histo_index, lbp_value);
}

int* compute_lbp_values(const unsigned char* image, const size_t width,
                        const size_t height)
{

    size_t sz;
    cudaDeviceGetLimit(&sz, cudaLimitPrintfFifoSize);
    //std::cout << "OOOOOOO: " << sz << std::endl;
    sz = 1048576 * 100;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

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
    // TODO: Essayer ce calcul
    //auto width_tiles = width / 16 + (width % 16 != 0);
    //auto height_tiles = height / 16 + (height % 16 != 0);
    auto tiles_number = width * height / 256;

    //printf("tiles_number: %d, our tiles: %d\n", tiles_number, width_tiles * height_tiles);

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
    //test<<<lbp_dim_grid, lbp_dim_block>>>();

    lbp_value_kernel<<<lbp_dim_grid, lbp_dim_block>>>(cuda_image, lbp_values, width, height, lbp_pitch);

    cudaDeviceSynchronize();

    //std::cout << "w " << w << " and h " << h << "\n";

    int* histo_tab;
    size_t histo_pitch;
    rc = cudaMallocPitch(&histo_tab, &histo_pitch, 256 * sizeof(int), tiles_number);
    if (rc)
    {
        std::cout << "Could not allocate memory for lbp values buffer\n";
        exit(EXIT_FAILURE);
    }

    cudaMemset2D(histo_tab, histo_pitch, 0, 256 * sizeof(int), tiles_number);

    cudaDeviceSynchronize();


    dim3 histo_dim_block(32, 8);
    dim3 histo_dim_grid(w, h);

    //std::cout << "lbp_pitch: " << lbp_pitch << ", " << "histo_pitch: " << histo_pitch << "\n";


    compute_histo_kernel<<<histo_dim_grid, histo_dim_block>>>(histo_tab, histo_pitch, lbp_values, lbp_pitch, tiles_number);

    cudaDeviceSynchronize();

    int* output = (int*) malloc(256 * tiles_number * sizeof(int));
    if (output == NULL)
    {
        std::cout << "CRINGE\n";
        exit(EXIT_FAILURE);
    }

    //cudaMemcpy(output, histo_tab + histo_pitch * 400 / sizeof(int), 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy2D(output, 256 * sizeof(int), histo_tab, histo_pitch, 256 * sizeof(int), tiles_number, cudaMemcpyDeviceToHost);

    //std::cout << "HISTO\n" << sizeof(unsigned char) << "\n";

    unsigned int sum = 0;
    for (auto i = 0; i < 256 * tiles_number; i++)
    {
       int value = *(output + i);
       //std::cout << i << ": " << value  << "\n";
       sum += value;
    }

    std::cout << "Sum: " << sum << "\n";

    assert(sum == 256 * tiles_number);

    //std::cout << "Tiles number " << tiles_number << '\n';

    cudaFree(cuda_image);
    cudaFree(lbp_values);
    cudaFree(histo_tab);

    return output;
}
