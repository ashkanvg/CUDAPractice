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


__global__ 
void histogram256(unsigned char* __restrict__ d_A, unsigned int* __restrict__ d_hist, int N){
    __shared__ unsigned int s_hist[256];
    size_t tid = threadIdx.x;

    // 1) use all threads to initiate shared memory
    for(size_t bin = tid; bin < 256; bin+= blockDim.x){
        s_hist[bin] = 0;
    }
    __syncthreads();

    // 2) update histogram
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < N; i+=stride){
        unsigned char v = d_A[i];
        atomicAdd(&s_hist[v], 1);
    }
    __syncthreads();


    // 3) merge shared memory histograms within blocks with single block
    for(int bin = tid; bin < 256; bin += blockDim.x){
        unsigned int count = s_hist[bin];
        if(count > 0){
            atomicAdd(&d_hist[bin], count);
        }
    }


}


int main(int argc, char **argv) {
    // data
    const int N = 1 << 20;  // 1,048,576 elements

    // ----- Host input -----
    std::vector<unsigned char> h_A(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<unsigned char>(i % 256); // nice uniform test
    }

    // Allocate memory on device
    unsigned char *d_A = nullptr; 
    unsigned int *d_hist = nullptr;
    cudaMalloc(&d_A, N * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    if(!d_A || !d_hist) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // Launch kernel 
    size_t threadsPerBlock = GPU_THREADS;
    size_t blocksPerGrid = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    blocksPerGrid = prop.multiProcessorCount * 8;
    if(blocksPerGrid * threadsPerBlock > N){
        blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    }
    if(blocksPerGrid == 0) blocksPerGrid = 1;

    histogram256<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    
    std::vector<unsigned int> h_hist(256);
    cudaMemcpy(h_hist.data(), d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);


    // ---- Print result ----
    printf("N = %d\n", N);
    printf("Histogram: \n");
    for (int i = 0; i < 256; ++i) printf("%d: %d\t", i+1, h_hist[i]);
    printf("\n");
    
    

    // Free memory
    cudaFree(d_A);
    cudaFree(d_hist);
    return EXIT_SUCCESS;
}