#ifndef IRGPUA_FIX_GPU_KK_CUH
#define IRGPUA_FIX_GPU_KK_CUH

#include "image.hh"

#define kernel_check_error()                        \
    {                                               \
        auto e = cudaGetLastError();                \
        if (e != cudaSuccess) {                     \
            std::string error = "Cuda failure in "; \
            error += __FILE__;                      \
            error += " at line ";                   \
            error += std::to_string(__LINE__);      \
            error += ": ";                          \
            error += cudaGetErrorString(e);         \
            exit(-1);                               \
        }                                           \
    }

__global__ void kk_mark_garbage(const int *buffer_in, int *buffer_out, int size);

__global__ void kk_exclusive_scan(int *buffer, int size);

__global__ void kk_scatter(int *to_fix, const int *predicate, int size);

__global__ void kk_correction(int *to_fix, int size);

__global__ void kk_fill_histo(int *histo, int *to_fix, int size);

__global__ void kk_inclusive_scan(int *buffer, int size);

__global__ void kk_find_first_non_zero(int *buffer, int size, int *result);

__global__ void kk_map_transform(int *to_fix, int *d_histo, int size, int *cdf_min);

void fix_image_gpu_kk(Image &to_fix);

#endif  // IRGPUA_FIX_GPU_KK_CUH
