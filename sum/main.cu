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
#define GPU_BLOCKS      256      // Should work with reduced threads/shared memory for cooperative launches
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

// GPU kernel: C[i] = A[i] + B[i]
// simple kernel style: one thread per element
__global__ void sumKernel(float *d_A, float *d_B, float *d_C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

// grid kernel style: one thread per block
__global__ void sumKernel_stride(const float* A, const float* B, float* C, int N) {
    int idx    = blockDim.x * blockIdx.x + threadIdx.x;   // starting index
    int stride = blockDim.x * gridDim.x;                  // jump size

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
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
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if(!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate memory.\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    // Read data from file
    for (int i = 0; i < N; i++) {fscanf(f, "%f", &h_A[i]);}
    for (int i = 0; i < N; i++) {fscanf(f, "%f", &h_B[i]);}


    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    if(!d_A || !d_B || !d_C) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        free(h_A);
        free(h_B);
        free(h_C);
        fclose(f);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel 
    size_t threadsPerBlock = GPU_THREADS;
    size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sumKernel_stride<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());          // check launch error
    CUDA_CHECK(cudaDeviceSynchronize());     // wait for GPU to finish

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));


    // Print result

    // ---- Print result ----
    printf("N = %d\n", N);
    printf("A: \t\t");
    for (int i = 0; i < N; ++i) printf("%g\t", h_A[i]);
    printf("\n");

    printf("B: \t\t");
    for (int i = 0; i < N; ++i) printf("%g\t", h_B[i]);
    printf("\n");

    printf("C = A + B: \t");
    for (int i = 0; i < N; ++i) printf("%g\t", h_C[i]);
    printf("\n");


    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    fclose(f);
    return EXIT_SUCCESS;
}