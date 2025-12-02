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

__global__ void blockScanExclusive(const float* d_in, float* d_out, float* d_block_sums, const int N){
    // Blelloch exclusive scan in shared memory:
    /*
    - Blelloch does this in two phases on an array sdata[0..n-1] in shared memory:
    - Upsweep (reduce): build a “sum tree”
    - Downsweep: walk back down the tree to turn sums into prefix sums
    - Think of it as:
        - First: compute partial sums up a binary tree.
        - Then: propagate prefix sums down the tree.
    */

    // Each block handles a chunk of 2 * blockDim.x elements:
    extern __shared__ float s_data[]; // size = 2 * blockDim.x

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    int blockStart = 2 * blockIdx.x * blockDim.x;

    int ai = blockStart + tid;
    int bi = blockStart + tid + blockDim.x;

    // Load up to 2*blockDim.x elements from global to shared
    /* So:
    - For a block with blockDim.x = 4, n = 2 * blockDim.x = 8.
    - Thread 0 loads indices 0 and 4 into sdata[0] and sdata[4].
    - Thread 1 loads indices 1 and 5 into sdata[1] and sdata[5].
    - etc.

    - Now we have a local array sdata[0..7] in shared memory to scan.
    */
    s_data[tid] = (ai < N) ? d_in[ai] : 0.0f;
    s_data[tid + blockDim.x] = (bi < N) ? d_in[bi] : 0.0f;

    __syncthreads();


    int n = 2 * blockDim.x; // e.g. n = 8 for blockDim.x = 4
    // reduce phase - Upsweep phase with small example (n = 8)
    /*
    sdata (initial) = [3, 1, 7, 0, 4, 1, 6, 3]
    index              0  1  2  3  4  5  6  7
    */
    int offset = 1;
    for(int d = n >> 1; d > 0; d>>=1){
        __syncthreads();
        if(tid < d){
            int i1 = offset * (2 * tid + 1) - 1;
            int i2 = offset * (2 * tid + 2) - 1;
            s_data[i2] += s_data[i1];            
        }
        offset<<=1;
    }
    /*
    offset = 1:
        tid=0:
        i1 = 1*(2*0+1)-1 = 0
        i2 = 1*(2*0+2)-1 = 1
        → sdata[1] += sdata[0] → 1 + 3 = 4

        tid=1:
        i1 = 1*(3)-1 = 2
        i2 = 1*(4)-1 = 3
        → sdata[3] += sdata[2] → 0 + 7 = 7

        tid=2:
        i1 = 1*(5)-1 = 4
        i2 = 1*(6)-1 = 5
        → sdata[5] += sdata[4] → 1 + 4 = 5

        tid=3:
        i1 = 1*(7)-1 = 6
        i2 = 1*(8)-1 = 7
        → sdata[7] += sdata[6] → 3 + 6 = 9    

    NEW S_DATA: 
    [_, 4, _, 7, _, 5, _, 9]
    [3, 4, 7, 7, 4, 5, 6, 9]

    Offset = 2:
        Upsweep iteration 2: d = 2, offset = 2

        Now:

        tid < 2 → tid = 0, 1

        Compute:

        tid=0:
        i1 = 2*(2*0+1)-1 = 2*1 - 1 = 1
        i2 = 2*(2*0+2)-1 = 2*2 - 1 = 3
        → sdata[3] += sdata[1] → 7 + 4 = 11

        tid=1:
        i1 = 2*(3)-1 = 5
        i2 = 2*(4)-1 = 7
        → sdata[7] += sdata[5] → 9 + 5 = 14

    NEW S_DATA:
    [_, 4, _,  7, _, 5, _,  9]
    [_, _, _, 11, _, _, _, 14]

    [3, 4, 7, 11, 4, 5, 6, 14]

    Offset = 4:
        Upsweep iteration 3: d = 1, offset = 4
        tid=0:
        i1 = 4*(1)-1 = 3
        i2 = 4*(2)-1 = 7
        → sdata[7] += sdata[3] → 14 + 11 = 25

    NEW S_DATA:
    [_, 4, _,  7, _, 5, _,  9]
    [_, _, _, 11, _, _, _, 14]
    [_, _, _, _, _, _, _,  25]

    [3, 4, 7, 11, 4, 5, 6, 25]

        sdata[7] = 25 is the sum of the whole array.
    */


    // s_data[n-1] now holds the sums of this blocks' chunk
    // write block sum to global memory
    if(tid == 0){
       d_block_sums[blockIdx.x] = s_data[n - 1];
       s_data[n-1] = 0.0f;                          // sdata = [3, 4, 7, 11, 4, 5, 6, 0]
    }

    // downsweep phase
    for(int d = 1; d < n; d <<= 1){
        offset >>= 1;
        __syncthreads();
        if(tid < d){
            int i1 = offset * (2*tid + 1) - 1;
            int i2 = offset * (2*tid + 2) - 1;
            float t = s_data[i1];
            s_data[i1] = s_data[i2];
            s_data[i2] += t; 
        }
    }
    __syncthreads();
    /*
    OFFSET = 8 (from previous loop)
        loop d = 1, 2, 4
        offset = 4, 2, 1

    Downsweep iteration 1: d = 1, offset becomes 4
        tid = 0:
        i1 = 4*(1) - 1 = 3
        i2 = 4*(2) - 1 = 7
        t = sdata[i1] = sdata[3] = 11
        sdata[3] = sdata[7] = 0
        sdata[7] += t → 0 + 11 = 11
    
    NEW S_DATA
    [_, _, _, 0, _, _, _, 11]
    [3, 4, 7, 0, 4, 5, 6, 11]


    Downsweep iteration 2: d = 2, offset becomes 2
    Now tid < 2 → tid = 0, 1.

        tid=0:
        i1 = 2*(1) - 1 = 1
        i2 = 2*(2) - 1 = 3
        t = sdata[1] = 4
        sdata[1] = sdata[3] = 0
        sdata[3] += t → 0 + 4 = 4

        tid=1:
        i1 = 2*(3) - 1 = 5
        i2 = 2*(4) - 1 = 7
        t = sdata[5] = 5
        sdata[5] = sdata[7] = 11
        sdata[7] += t → 11 + 5 = 16

    NEW S_DATA
    [3, 4, 7, 0, 4, 5,  6, 11]

    [_, _, _, 0, _, _,  _, 11]
    [_, 0, _, 4, _, 11, _, 16]

    [3, 0, 7, 4, 4, 11, 6, 16]


    Downsweep iteration 3: d = 4, offset becomes 1
    Now tid < 4 → tid = 0,1,2,3.

        For each:

        tid=0:
        i1 = 1*(1) - 1 = 0
        i2 = 1*(2) - 1 = 1
        t = sdata[0] = 3
        sdata[0] = sdata[1] = 0
        sdata[1] += t → 0 + 3 = 3

        tid=1:
        i1 = 1*(3) - 1 = 2
        i2 = 1*(4) - 1 = 3
        t = sdata[2] = 7
        sdata[2] = sdata[3] = 4
        sdata[3] += t → 4 + 7 = 11

        tid=2:
        i1 = 1*(5) - 1 = 4
        i2 = 1*(6) - 1 = 5
        t = sdata[4] = 4
        sdata[4] = sdata[5] = 11
        sdata[5] += t → 11 + 4 = 15

        tid=3:
        i1 = 1*(7) - 1 = 6
        i2 = 1*(8) - 1 = 7
        t = sdata[6] = 6
        sdata[6] = sdata[7] = 16
        sdata[7] += t → 16 + 6 = 22  


    NEW S_DATA
    [3, 4, 7,  0,  4,  5,  6, 11]
    [_, _, _,  0,  _,  _,  _, 11]
    [_, 0, _,  4,  _, 11,  _, 16]
    [0, 3, 4, 11, 11, 15, 16, 22]
    */


    // now s_data[] holds the exclusive scan of the block's chunk
    if(ai < N) d_out[ai] = s_data[tid];
    if(bi < N) d_out[bi] = s_data[tid + blockDim.x];

}

void exclusive_scan_host(std::vector<float>& arr) {
    float running = 0.0f;
    for (size_t i = 0; i < arr.size(); ++i) {
        float tmp = arr[i];
        arr[i] = running;
        running += tmp;
    }
}

__global__ void addBlockOffset(const float* d_in, float* out, float* d_block_sums, const int N){
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += stride){
        int block_segment = i / (2 * blockDim.x);
        float offset = d_block_sums[block_segment];
        d_out[i] += offset;
    }
}

int main(int argc, char **argv) {
    const int N = 1 << 20;  // 1,048,576 elements

    // Host input
    std::vector<float> h_in(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;  // easy to check: prefix sum should be i
    }

    // Allocate memory on device
    float *d_in = nullptr, d_out = nullptr, *d_block_sums = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    if(!d_in || !d_out) {
        fprintf(stderr, "Failed to allocate memory on device.\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel 
    // step 0: 
    size_t threadsPerBlock = GPU_THREADS;
    size_t blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
    size_t elementsPerBlock = 2 * threadsPerBlock;

    // step 1:
    size_t shmem_bytes = elementsPerBlock*sizeof(float);
    blockScanExclusive<<<blocksPerGrid, threadsPerBlock, elementsPerBlock>>>(d_in, d_out, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // step 2:
    std::vector<float> h_block_sums(blocksPerGrid);
    cudaMemcpy(h_block_sums.data(), d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    exclusive_scan_host(h_block_sums);
    cudaMemcpy(d_block_sums, h_block_sums.data(), blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice);


    // step 3: 
    addBlockOffset<<<blocksPerGrid, threadsPerBlock, elementsPerBlock>>>(d_in, d_out, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    
    // step 4:
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    
    
    // Print result Validation
    for(int i = 0; i < N; i++){
        if(h_out[i] != i){
            printf("Validation failed at index %d\n", i);
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(d_block_sums);
            return EXIT_FAILURE;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    return EXIT_SUCCESS;
}