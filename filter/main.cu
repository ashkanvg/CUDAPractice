#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>


// Thrust for exclusive_scan on the GPU
#include <thrust/device_ptr.h>
#include <thrust/scan.h>


#define WARP_SZ         32
#define MAX_GPU_BLOCKS  (WARP_SZ*8)
#define GPU_BLOCKS      28      // Should work with reduced threads/shared memory for cooperative launches
#define GPU_THREADS     1024    // Must match GCD of maxThreadsPerBlock and maxThreadsPerMultiProcessor
#define GPU_SHM_BYTES   33024   // Set to max - query will find the actual usable amount, we just need >= that
#define STREAM_COUNT    1      // number of stream is 32

#define CUDA_CHECK(err)                                                       \
    do {                                                                      \
        cudaError_t err__ = (err);                                            \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)



static_assert(MAX_GPU_BLOCKS >= GPU_BLOCKS && 0 == (MAX_GPU_BLOCKS % WARP_SZ), "");
static_assert(WARP_SZ == 32, "Yeah... this is assumed in a lot of places.");
static_assert(GPU_THREADS >= MAX_GPU_BLOCKS, "o.w. gbl scans require a loop");


// Sum 'val' across the 32 threads of a warp
__inline__ __device__
float warpReduceSum(float val) {
    // Full mask: all 32 lanes active
    unsigned int mask = 0xffffffff;
    // Tree reduction: add values from threads offset away
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}


__global__ 
void mark_flag(const int* d_A, int* d_flag, int N){
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = idx; i < N; i+=stride){
        d_flag[i] = (d_A[i] % 2 == 0) ? 1 : 0;
    }
}

__global__ 
void scatter(const int* d_A, int* d_flag, int* d_indices, int* d_output, int N){
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = idx; i < N && d_flag[i]; i+=stride){
        int new_idx = d_indices[i];
        d_output[new_idx] = d_A[i];
    }
}


int main(int argc, char **argv) {
    const char* fileName = "sample.txt";
    // Read input from file
    FILE* f = fopen(fileName, "r");
    if (!f) {
        fprintf(stderr, "Could not open input file: %s\n", fileName);
        return EXIT_FAILURE;
    }

    // Read N from file
    int N;
    if (fscanf(f, "%d", &N) != 1 || N <= 0) {
        fprintf(stderr, "Failed to read valid N from file.\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    // Allocate memory on host
    int *h_A = (int *)malloc(N * sizeof(int));
    if(!h_A) {
        fprintf(stderr, "Failed to allocate memory.\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    // Read data from file
    for (int i = 0; i < N; i++) {fscanf(f, "%d", &h_A[i]);}


    // Allocate memory on device
    int *d_A = nullptr, *d_flag = nullptr, *d_indices = nullptr, *d_output = nullptr;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_flag, N * sizeof(int));
    cudaMalloc(&d_indices, N* sizeof(int));
    cudaMalloc(&d_output, N*sizeof(int));
    if(!d_A || !d_flag) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        free(h_A);
        fclose(f);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel 
    size_t threadsPerBlock = GPU_THREADS;
    size_t blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;

    mark_flag<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_flag, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        thrust::device_ptr<int> d_flag_ptr(d_flag);
        thrust::device_ptr<int> d_indices_ptr(d_indices);
        thrust::exclusive_scan(d_flag_ptr, d_flag_ptr + N, d_indices_ptr);
    }

    
    int h_last_flag = 0, h_last_index = 0;
    cudaMemcpy(&h_last_flag, d_flag + (N-1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_index, d_indices + (N-1), sizeof(int), cudaMemcpyDeviceToHost);
    int new_array_size = h_last_flag + h_last_index;



    scatter<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_flag, d_indices, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    std::vector<int> h_output(new_array_size);
    cudaMemcpy(h_output.data(), d_output, new_array_size * sizeof(int), cudaMemcpyDeviceToHost);


    // ---- Print result ----
    printf("N = %d\n", N);
    printf("A: \t\t");
    for (int i = 0; i < N; ++i) printf("%d\t", h_A[i]);
    printf("\n");

    printf("N' (new_array_size) = %d\n", new_array_size);
    printf("EVENS: \t\t");
    for (int i = 0; i < new_array_size; ++i) printf("%d\t", h_output[i]);
    printf("\n");
    
    

    // Free memory
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_flag);
    cudaFree(d_indices);
    cudaFree(d_output);
    fclose(f);
    return EXIT_SUCCESS;
}