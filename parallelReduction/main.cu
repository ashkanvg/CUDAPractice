#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>


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



__global__ void reduceKernel(float *d_A, float *d_block_sums, int N) {
    extern __shared__ float s_data[]; // shared memory buffer

     // global thread index
     size_t tid  = threadIdx.x;
     size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
     size_t grid = blockDim.x * gridDim.x;

    // 1) Each thread computes a local sum over a grid-stride loop
    float local_sum = 0.0f;
    for (size_t i = idx; i < N; i += grid) {
        local_sum += d_A[i];
    }

    // 2) Store in shared memory
    s_data[tid] = local_sum;
    __syncthreads();


    // 3) Reduction: reduce local sums to a single value
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 4) Write result to output array
    if (tid == 0) {
        d_block_sums[blockIdx.x] = s_data[0];
    }
}

float reduceWrapper(float *d_A, int N) {
    size_t threadsPerBlock = GPU_THREADS;
    size_t blocksPerGrid = 0;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    blocksPerGrid = prop.multiProcessorCount * 8;

    if(blocksPerGrid * threadsPerBlock > N) {
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    }


    // per block sum:
    float *d_block_sums = nullptr; // per each block, we need to store the sum
    cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));
    if(!d_block_sums) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        return EXIT_FAILURE;
    }

    size_t sharedMemPerBlock = threadsPerBlock * sizeof(float);
    // size_t sharedMemPerBlock = sharedMem / blockCount;

    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemPerBlock>>>(d_A, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());          // check launch error
    CUDA_CHECK(cudaDeviceSynchronize());     // wait for GPU to finish

    // now we have 'blocks' partial sums in d_block_sums
    // if blocks is small, copy back and finish on CPU
    std::vector<float> h_block_sums(blocksPerGrid);
    cudaMemcpy(h_block_sums.data(), d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_block_sums);


    // final sum on CPU
    float total = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        total += h_block_sums[i];
    }

    return total;
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
    float *h_A = (float *)malloc(N * sizeof(float));
    if(!h_A) {
        fprintf(stderr, "Failed to allocate memory.\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    // Read data from file
    for (int i = 0; i < N; i++) {fscanf(f, "%f", &h_A[i]);}


    // Allocate memory on device
    float *d_A;
    cudaMalloc(&d_A, N * sizeof(float));
    if(!d_A) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        free(h_A);
        fclose(f);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel 
    float total = reduceWrapper(d_A, N);


    // Print result

    // ---- Print result ----
    printf("N = %d\n", N);
    printf("A: \t\t");
    for (int i = 0; i < N; ++i) printf("%g\t", h_A[i]);
    printf("\n");


    printf("Total: %g\t\t\n", total);

    // Free memory
    free(h_A);
    cudaFree(d_A);
    fclose(f);
    return EXIT_SUCCESS;
}