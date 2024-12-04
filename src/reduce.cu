#include <cuda_profiler_api.h>

#include "reduce.cuh"

#define BLOCKSIZE 128

__global__ void crappy_reduce(int *buffer, int *total, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    atomicAdd(&total[0], buffer[idx]);
}

int crappy_reduce_gpu(Image image, int image_size) {
    int *d_reduce;
    int *d_total;
    int h_total = 0;
    const int block_size = 64;
    const int gridSize = (image_size + block_size - 1) / block_size;

    cudaMalloc(&d_reduce, image_size * sizeof(int));
    cudaMalloc(&d_total, sizeof(int));
    cudaMemcpy(d_reduce, image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total, &h_total, sizeof(int), cudaMemcpyHostToDevice);

    crappy_reduce<<<gridSize, block_size>>>(d_reduce, d_total, image_size);

    cudaStreamSynchronize(0);
    cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_reduce);
    cudaFree(d_total);
    return h_total;
}

template <int BLOCK_SIZE>
__device__ void generic_warp_reduce(int *s_array, int tid) {
    if constexpr (BLOCK_SIZE >= 64) {
        s_array[tid] += s_array[tid + 32];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 32) {
        s_array[tid] += s_array[tid + 16];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 16) {
        s_array[tid] += s_array[tid + 8];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 8) {
        s_array[tid] += s_array[tid + 4];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 4) {
        s_array[tid] += s_array[tid + 2];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 2) {
        s_array[tid] += s_array[tid + 1];
        __syncwarp();
    }
}

template <typename T, int BLOCK_SIZE>
__global__ void kernel_adapt_warp_reduce(const T *__restrict__ buffer, T *__restrict__ total, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > size) return;

    __shared__ int s_array[BLOCK_SIZE];
    s_array[threadIdx.x] = 0;
    int globIdx = idx;

#pragma unroll
    for (int i = 0; globIdx < size; i++) {
        s_array[threadIdx.x] += buffer[globIdx];
        globIdx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    if constexpr (BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) s_array[threadIdx.x] += s_array[threadIdx.x + 256];
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) s_array[threadIdx.x] += s_array[threadIdx.x + 128];
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 128) {
        if (threadIdx.x < 64) s_array[threadIdx.x] += s_array[threadIdx.x + 64];
        __syncthreads();
    }

    if (threadIdx.x < 32) generic_warp_reduce<BLOCK_SIZE>(s_array, threadIdx.x);

    if (threadIdx.x == 0) atomicAdd(&total[0], s_array[0]);
}

int adapt_warp_reduce(Image image, int image_size) {
    int *d_reduce;
    int *d_total;
    int h_total = 0;

    const int work_per_thread = 8;
    const int gridSize = ((image_size + BLOCKSIZE - 1) / BLOCKSIZE) / work_per_thread;

    cudaMalloc(&d_reduce, image_size * sizeof(int));
    cudaMalloc(&d_total, sizeof(int));
    cudaMemcpy(d_reduce, image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total, &h_total, sizeof(int), cudaMemcpyHostToDevice);

    kernel_adapt_warp_reduce<int, BLOCKSIZE><<<gridSize, BLOCKSIZE>>>(d_reduce, d_total, image_size);

    cudaStreamSynchronize(0);
    cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_reduce);
    cudaFree(d_total);
    return h_total;
}

__device__ int register_warp_reduce(int sum) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_down_sync(~0, sum, offset);
    return sum;
}

__global__ void kernel_register_warp_reduce(int *buffer, int *total, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    int sum = 0;
#pragma unroll
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) sum += buffer[idx];

    sum = register_warp_reduce(sum);

    if (threadIdx.x % 32 == 0) atomicAdd(&total[0], sum);
}

int register_warp_reduce(Image image, int image_size) {
    int *d_reduce;
    int *d_total;
    int h_total = 0;
    const int work_per_thread = 1;
    const int blockSize = 64;
    const int gridSize = ((image_size + blockSize - 1) / blockSize) / work_per_thread;

    cudaMalloc(&d_reduce, image_size * sizeof(int));
    cudaMalloc(&d_total, sizeof(int));
    cudaMemcpy(d_reduce, image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total, &h_total, sizeof(int), cudaMemcpyHostToDevice);

    kernel_register_warp_reduce<<<gridSize, blockSize>>>(d_reduce, d_total, image_size);

    cudaStreamSynchronize(0);
    cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_reduce);
    cudaFree(d_total);
    return h_total;
}
