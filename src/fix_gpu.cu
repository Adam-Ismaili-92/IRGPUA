#include <cuda/atomic>

#include "fix_gpu.cuh"

#define HIST_SIZE 256

struct flag_value {
    char flag;
    int value;
};

template <int BLOCK_SIZE>
__global__ void mark_and_scan(const int *to_fix, int *predicate, int size, cuda::std::atomic<flag_value> *flags, int *id) {
    __shared__ int s_array[BLOCK_SIZE + 1];
    __shared__ int s_thread0_var;
    constexpr int garbage_val = -27;

    if (threadIdx.x == 0) s_thread0_var = atomicAdd(id, 1);
    __syncthreads();
    int blockId = s_thread0_var;
    int idx = blockId * BLOCK_SIZE + threadIdx.x;
    // Padding with 0 when not a power of 2
    s_array[threadIdx.x] = (idx >= size || to_fix[idx] == garbage_val) ? 0 : 1;
    __syncthreads();
    // SKLANSKY SCAN
    for (int i = 1; i < BLOCK_SIZE; i *= 2) {
        if ((threadIdx.x % (i * 2)) == (i * 2 - 1)) {
            s_array[threadIdx.x] += s_array[threadIdx.x - i];
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_SIZE - 1) {
        int lastValue = s_array[threadIdx.x];
        s_array[threadIdx.x] = 0;
        s_array[BLOCK_SIZE] = lastValue;
        flags[blockId].store({'A', s_array[BLOCK_SIZE]});
    }
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2) {
        if ((threadIdx.x % (i * 2)) == (i * 2 - 1)) {
            int tmp = s_array[threadIdx.x - i];
            s_array[threadIdx.x - i] = s_array[threadIdx.x];
            s_array[threadIdx.x] += tmp;
        }
        __syncthreads();
    }
    if (idx >= size) return;
    predicate[idx] = s_array[threadIdx.x + 1];
    // SCAN END
    bool no_back_sum = true;
    while (blockId != 0 && threadIdx.x == 0 && no_back_sum) {
        int sum = 0;
        for (int i = blockId - 1; i >= 0; --i) {
            flag_value flag = flags[i].load();
            if (flag.flag == 0) break;
            sum += flag.value;
            if (flag.flag == 'P' || i == 0) {
                no_back_sum = false;
                s_thread0_var = sum;
                break;
            }
        }
    }
    __syncthreads();
    if (blockId != 0) predicate[idx] += s_thread0_var;

    if (threadIdx.x == 0) flags[blockId].store({'P', s_array[BLOCK_SIZE] + s_thread0_var});
}

__global__ void scatter(const int *to_fix, int *to_fix_fixed, const int *predicate, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    constexpr int garbage_val = -27;

    if (to_fix[idx] != garbage_val) to_fix_fixed[predicate[idx]] = to_fix[idx];
}

__global__ void correction_and_histo(int4 *to_fix, int *histo, int size, int size_original) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    to_fix[idx].x += 1;
    to_fix[idx].y -= 5;
    to_fix[idx].z += 3;
    to_fix[idx].w -= 8;

    __shared__ int s_histo[HIST_SIZE];
    if (threadIdx.x == 0)
        for (int i = 0; i < HIST_SIZE; i += 1) s_histo[i] = 0;
    __syncthreads();

    if (idx < size - 1) {
        atomicAdd_block(&s_histo[to_fix[idx].x], 1);
        atomicAdd_block(&s_histo[to_fix[idx].y], 1);
        atomicAdd_block(&s_histo[to_fix[idx].z], 1);
        atomicAdd_block(&s_histo[to_fix[idx].w], 1);
    } else {
        int mod4 = size_original % 4;
        if (mod4 > 0) atomicAdd_block(&s_histo[to_fix[idx].x], 1);
        if (mod4 > 1) atomicAdd_block(&s_histo[to_fix[idx].y], 1);
        if (mod4 > 2) atomicAdd_block(&s_histo[to_fix[idx].z], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0)
        for (int i = 0; i < HIST_SIZE; i += 1) atomicAdd_system(&histo[i], s_histo[i]);
}

__global__ void inclusive_scan_and_find_first_non_zero(int *histo, int *non_zero) {
    __shared__ int s_array[HIST_SIZE];

    s_array[threadIdx.x] = histo[threadIdx.x];
    __syncthreads();

#pragma unroll
    for (int i = 1; i < HIST_SIZE; i *= 2) {
        int tmp = s_array[threadIdx.x];
        __syncthreads();
        if (threadIdx.x + i < HIST_SIZE) s_array[threadIdx.x + i] += tmp;
        __syncthreads();
    }
    histo[threadIdx.x] = s_array[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < HIST_SIZE; ++i) {
            if (histo[i] != 0) {
                *non_zero = i;
                return;
            }
        }
    }
}

__global__ void map_transform(int *to_fix, const int *d_histo, int size, const int *cdf_min_idx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    int cdf_min = d_histo[cdf_min_idx[0]];

    float scaled = (static_cast<float>(d_histo[to_fix[idx]] - cdf_min) / static_cast<float>(size - cdf_min)) * 255.0f;
    to_fix[idx] = static_cast<int>(llroundf(scaled));
}

void fix_image_gpu(Image &to_fix) {
    int *d_to_fix;
    int *d_to_fix_correct_size;

    int *d_predicate;
    cuda::std::atomic<flag_value> *d_flags;
    int *d_id;

    int *d_histo;
    int *d_non_zero;

    constexpr int block_size = 64;
    constexpr int mark_block_size = 1024;
    constexpr int correction_block_size = 128;

    const int gridSize = (to_fix.size() + block_size - 1) / block_size;
    const int image_size = to_fix.width * to_fix.height;
    const int gridSize_post_scatter = (image_size + block_size - 1) / block_size;

    cudaMalloc(&d_to_fix, to_fix.size() * sizeof(int));
    cudaMalloc(&d_predicate, (to_fix.size() + 1) * sizeof(int));
    cudaMalloc(&d_flags, gridSize * sizeof(flag_value));
    cudaMalloc(&d_id, sizeof(int));

    cudaMemcpy(d_to_fix, to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_predicate, 0, (to_fix.size() + 1) * sizeof(int));

    cudaMemset(d_flags, 0, gridSize * sizeof(flag_value));
    cudaMemset(d_id, 0, sizeof(int));

    // Discard garbage values
    mark_and_scan<mark_block_size>
        <<<(to_fix.size() + mark_block_size - 1) / mark_block_size, mark_block_size>>>(d_to_fix, d_predicate + 1, to_fix.size(), d_flags, d_id);
    cudaFree(d_flags);  // Last usage of d_flags
    cudaFree(d_id);     // Last usage of d_id

    cudaMalloc(&d_to_fix_correct_size, (((image_size / 4) + 1) * 4) * sizeof(int));
    scatter<<<gridSize, block_size>>>(d_to_fix, d_to_fix_correct_size, d_predicate, to_fix.size());
    cudaFree(d_predicate);  // Last usage of d_predicate
    cudaFree(d_to_fix);     // Last usage of d_to_fix

    cudaMalloc(&d_histo, HIST_SIZE * sizeof(int));
    cudaMalloc(&d_non_zero, sizeof(int));
    cudaMemset(d_histo, 0, HIST_SIZE * sizeof(int));
    cudaMemset(d_non_zero, 127, sizeof(int));

    // Correct injected values and Histogram
    correction_and_histo<<<(image_size + correction_block_size - 1) / correction_block_size, correction_block_size>>>(
        reinterpret_cast<int4 *>(d_to_fix_correct_size), d_histo, image_size / 4 + 1, image_size);

    inclusive_scan_and_find_first_non_zero<<<1, HIST_SIZE>>>(d_histo, d_non_zero);
    map_transform<<<gridSize_post_scatter, block_size>>>(d_to_fix_correct_size, d_histo, image_size, d_non_zero);
    cudaFree(d_histo);     // Last usage of d_histo
    cudaFree(d_non_zero);  // Last usage of d_first_non_zero

    cudaMemcpy(to_fix.buffer, d_to_fix_correct_size, image_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_to_fix_correct_size);
    cudaStreamSynchronize(0);
}
