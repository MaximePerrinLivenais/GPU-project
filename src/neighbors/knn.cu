#include "knn.cuh"

#include <iostream>

#define N_CLUSTERS = 16
#define TILE_SIZE = 256 // A MOVE DANS UN HH

__global__ void compute_nearest_neighbors(const int* histo_tab,
                                        const size_t histo_pitch,
                                        const float* clusters,
                                        const size_t cluster_pitch,
                                        int* results,
                                        const size_t tiles_number)
{
    size_t x = threadIdx.x;
    size_t tile_index = blockIdx.x;

    if (x >= 256 || tile_index >= tiles_number)
        return;

    __shared__ float cluster_distances[16];

    for (auto y = 0; x == 0 && y < 16; y++)
        cluster_distances[y] = 0;

    __syncthreads();

    int value = *(histo_tab + tile_index * histo_pitch / sizeof(int) + x);

    for (auto y = 0; y < 16; y++)
    {
        float cluster_value = *(clusters + y * cluster_pitch / sizeof(float) + x);

        // Euclidean distance
        float local_distance = (cluster_value - value) * (cluster_value - value);

        atomicAdd(cluster_distances + y, local_distance);
    }

    __syncthreads();

    if (x != 0)
        return;

    //printf("cluster_distance[0]: %f\n", cluster_distances[0]);

    auto result_ptr = results + tile_index;
    *result_ptr = 0;
    for (int i = 1; i < 16; i++)
    {
        if (cluster_distances[*result_ptr] > cluster_distances[i])
        {
            //printf("Change\n");
            *result_ptr = i;
        }
        //printf("clusters_distance: %f at %d\n", cluster_distances[i], i);
    }

    //printf("result: %d\n", *result_ptr);
}

void k_nearest_neighbors(const int* histo_tab, const float* clusters, const size_t tiles_number)
{

    /*for (int i = 0; i < 16; ++i)
        printf("%f|", clusters[i * 256], i);

    std::cout << "\n";

    */
    printf("histo[1]: %d\n", histo_tab[2]);
    std::cout << "\n";

    size_t sz;
    cudaDeviceGetLimit(&sz, cudaLimitPrintfFifoSize);

    sz = 1048576 * 100;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

    cudaError_t rc = cudaSuccess;

    int* cuda_histo_tab;
    size_t cuda_histo_tab_pitch;
    auto tile_byte_size = 256 * sizeof(int);

    rc = cudaMallocPitch(&cuda_histo_tab, &cuda_histo_tab_pitch, tile_byte_size, tiles_number);
    if (rc)
    {
        std::cout << "Could not allocate memory for tiles histogram"
            << "on the device when computing nearest_neighbors\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy2D(cuda_histo_tab, cuda_histo_tab_pitch, histo_tab, tile_byte_size,
                        tile_byte_size, tiles_number, cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy memory for tiles histogram"
            << "on the device when computing nearest_neighbors\n";
        exit(EXIT_FAILURE);
    }


    float* cuda_clusters;
    size_t cuda_clusters_pitch;
    auto cluster_byte_size = 256 * sizeof(float);

    rc = cudaMallocPitch(&cuda_clusters, &cuda_clusters_pitch, cluster_byte_size, 16);
    if (rc)
    {
        std::cout << "Could not allocate memory for clusters"
            << "on the device when computing nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    rc = cudaMemcpy2D(cuda_clusters, cuda_clusters_pitch, clusters, cluster_byte_size,
                        cluster_byte_size, 16, cudaMemcpyHostToDevice);
    if (rc)
    {
        std::cout << "Could not copy memory for clusters"
            << "on the device when computing nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    int* result;

    rc = cudaMalloc(&result, sizeof(int) * tiles_number);
    if (rc)
    {
        std::cout << "Could not allocate memory for nearest neighbors result"
            << "on the device when computing nearest neighbors\n";
        exit(EXIT_FAILURE);
    }

    dim3 block_dim(256, 16);
    compute_nearest_neighbors<<<tiles_number, 256>>>(cuda_histo_tab, cuda_histo_tab_pitch,
        cuda_clusters, cuda_clusters_pitch, result, tiles_number);


    int* output = (int*) malloc(sizeof(int) * tiles_number);
    rc = cudaMemcpy(output, result, sizeof(int) * tiles_number, cudaMemcpyDeviceToHost);


    for (int i = 0; i < tiles_number; i++)
    {
        printf("value: %d at %d\n", output[i], i);
    }

    std::free(output);

    cudaFree(cuda_clusters);
    cudaFree(result);
}
