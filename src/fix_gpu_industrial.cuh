#ifndef IRGPUA_FIX_GPU_INDUSTRIAL_CUH
#define IRGPUA_FIX_GPU_INDUSTRIAL_CUH

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include "image.hh"

// unary thrust function to replace -27 by 1

struct is_neg_27 : public thrust::unary_function<int, int> {
    __device__ int operator()(int val) const { return (val != -27) ? 1 : 0; }
};

// unary thrust function to replace val if garbage

struct correction : public thrust::unary_function<int, int> {
    __device__ int operator()(int index, int to_fix_val) {
        switch (index % 4) {
            case 0:
                return to_fix_val + 1;

            case 1:
                return to_fix_val - 5;

            case 2:
                return to_fix_val + 3;

            default:
                return to_fix_val - 8;
        }
    }
};

// unary thrust function to map transform

struct map_transform {
    int image_size;
    float cdf_min;
    thrust::device_ptr<int> histo;

    map_transform(int _image_size, int _cdf_min, thrust::device_ptr<int> _histo) : image_size(_image_size), cdf_min(_cdf_min), histo(_histo) {}

    __device__ int operator()(int& pixel) {
        return static_cast<int>(llroundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f));
    }
};

struct HistogramFunctor {
    int* d_histo;

    HistogramFunctor(int* _d_histo) : d_histo(_d_histo) {}

    __device__ void operator()(const unsigned char& pixelValue) { atomicAdd(&(d_histo[pixelValue]), 1); }
};

void fix_image_gpu_industrial(Image& to_fix);

#endif  // IRGPUA_FIX_GPU_INDUSTRIAL_CUH
