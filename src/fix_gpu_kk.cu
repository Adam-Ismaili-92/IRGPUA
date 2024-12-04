#include <assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "fix_gpu_kk.cuh"

__global__ void kk_mark_garbage(const int *buffer_in, int *buffer_out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    constexpr int garbage_val = -27;
    if (buffer_in[idx] != garbage_val) buffer_out[idx] = 1;
}

__global__ void kk_exclusive_scan(int *buffer, int size) {
    int acc = 0;
    int tmp;
    for (int i = 0; i < size; ++i) {
        tmp = buffer[i];
        buffer[i] = acc;
        acc += tmp;
    }
}

__global__ void kk_scatter(int *to_fix, const int *predicate, int size) {
    constexpr int garbage_val = -27;
    for (int i = 0; i < size; ++i) {
        if (to_fix[i] != garbage_val) to_fix[predicate[i]] = to_fix[i];
    }
}

__global__ void kk_correction(int *to_fix, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    switch (idx % 4) {
        case 0:
            to_fix[idx] += 1;
            break;
        case 1:
            to_fix[idx] -= 5;
            break;
        case 2:
            to_fix[idx] += 3;
            break;
        default:
            to_fix[idx] -= 8;
    }
}

__global__ void kk_fill_histo(int *histo, int *to_fix, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    atomicAdd(&histo[to_fix[idx]], 1);
}

__global__ void kk_inclusive_scan(int *buffer, int size) {
    for (int i = 1; i < size; ++i) {
        buffer[i] += buffer[i - 1];
    }
}

__global__ void kk_find_first_non_zero(int *buffer, int size, int *result) {
    bool is_found = false;
    int i = 0;

    while (i < size && !is_found) {
        if (buffer[i] != 0)
            is_found = true;
        else
            i++;
    }

    result[0] = buffer[i];
}

__global__ void kk_map_transform(int *to_fix, int *d_histo, int size, int *cdf_min) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    float scaled = (static_cast<float>(d_histo[to_fix[idx]] - *cdf_min) / static_cast<float>(size - *cdf_min)) * 255.0f;
    to_fix[idx] = static_cast<int>(llroundf(scaled));
}

void fix_image_gpu_kk(Image &to_fix) {
    int *d_to_fix;
    int *d_predicate;
    int *d_histo;
    int *d_first_non_zero;

    const int block_size = 64;
    const int gridSize = (to_fix.size() + block_size - 1) / block_size;
    const int image_size = to_fix.width * to_fix.height;

    cudaMalloc(&d_to_fix, to_fix.size() * sizeof(int));
    cudaMalloc(&d_predicate, to_fix.size() * sizeof(int));
    cudaMalloc(&d_histo, 256 * sizeof(int));
    cudaMalloc(&d_first_non_zero, sizeof(int));

    cudaMemcpy(d_to_fix, to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_predicate, 0, to_fix.size() * sizeof(int));
    cudaMemset(d_histo, 0, 256 * sizeof(int));

    // Discard garbage values
    kk_mark_garbage<<<gridSize, block_size>>>(d_to_fix, d_predicate, to_fix.size());
    kk_exclusive_scan<<<1, 1>>>(d_predicate, to_fix.size());
    kk_scatter<<<1, 1>>>(d_to_fix, d_predicate, to_fix.size());
    cudaFree(d_predicate);  // Last usage of d_predicate

    // Correct injected values
    kk_correction<<<gridSize, block_size>>>(d_to_fix, image_size);

    // Histogram
    kk_fill_histo<<<gridSize, block_size>>>(d_histo, d_to_fix, image_size);
    kk_inclusive_scan<<<1, 1>>>(d_histo, 256);
    kk_find_first_non_zero<<<1, 1>>>(d_histo, 256, d_first_non_zero);
    kk_map_transform<<<gridSize, block_size>>>(d_to_fix, d_histo, image_size, d_first_non_zero);
    cudaFree(d_histo);           // Last usage of d_histo
    cudaFree(d_first_non_zero);  // Last usage of d_first_non_zero

    cudaMemcpy(to_fix.buffer, d_to_fix, to_fix.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_to_fix);
    cudaStreamSynchronize(0);
}
