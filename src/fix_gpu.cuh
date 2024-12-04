#ifndef IRGPUA_FIX_GPU_CUH
#define IRGPUA_FIX_GPU_CUH

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

void fix_image_gpu(Image &to_fix);

#endif  // IRGPUA_FIX_GPU_CUH
