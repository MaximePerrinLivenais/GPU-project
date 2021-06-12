#include "lbp.cuh"

#include <cassert>
#include <stdio.h>

__global__ void lbp_value_kernel(const unsigned char* image,
                                    unsigned char* lbp_values,
                                    const int width,
                                    const int height,
                                    const size_t pitch)
{
    #ifdef __CUDA_ARCH__
        printf("CUDA architecture of the current running device code: %d.\n", __CUDA_ARCH__);
        printf("Image height: %d. Image width: %d.\n", height, width);
    #endif
}

void compute_lbp_values(const unsigned char *image, const size_t width, const size_t height)
{
    cudaError_t rc = cudaSuccess;

    unsigned char* lbp_values;
    size_t pitch;

    rc = cudaMallocPitch(&lbp_values, &pitch, width * sizeof(char), height);
    if (rc)
        exit(1);

    lbp_value_kernel<<<1,1>>>(image, lbp_values, width, height, pitch);

    cudaDeviceSynchronize();
}
