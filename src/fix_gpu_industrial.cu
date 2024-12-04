#include "fix_gpu_industrial.cuh"

void fix_image_gpu_industrial(Image& to_fix) {
    thrust::device_vector<int> d_to_fix(to_fix.size());
    thrust::device_vector<int> d_predicate(to_fix.size(), 0);
    thrust::device_vector<int> d_histo(256, 0);
    thrust::device_vector<int>::iterator d_first_non_zero;

    int cdf_min = 0;
    const int image_size = to_fix.width * to_fix.height;

    thrust::copy(to_fix.buffer, to_fix.buffer + to_fix.size(), d_to_fix.begin());

    // Discard garbage values
    thrust::transform(thrust::device, d_to_fix.begin(), d_to_fix.end(), d_predicate.begin(), is_neg_27());

    thrust::exclusive_scan(thrust::device, d_predicate.begin(), d_predicate.end(), d_predicate.begin(), 0);

    thrust::device_vector<int> output_correction(image_size);

    thrust::scatter_if(thrust::device, d_to_fix.begin(), d_to_fix.end(), d_predicate.begin(), d_to_fix.begin(), output_correction.begin(),
                       thrust::placeholders::_1 != -27);

    // Correct injected values
    thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(image_size), output_correction.begin(),
                      output_correction.begin(), correction());

    thrust::copy(output_correction.begin(), output_correction.end(), d_to_fix.begin());
    // last use of output_correction

    // Histogram
    thrust::for_each(d_to_fix.begin(), d_to_fix.begin() + image_size, HistogramFunctor(thrust::raw_pointer_cast(d_histo.data())));

    thrust::inclusive_scan(thrust::device, d_histo.begin(), d_histo.end(), d_histo.begin());

    d_first_non_zero = thrust::find_if(d_histo.begin(), d_histo.end(), [] __device__(int v) { return v != 0; });

    if (d_first_non_zero != d_histo.end()) cdf_min = *d_first_non_zero;

    thrust::transform(d_to_fix.begin(), d_to_fix.begin() + image_size, d_to_fix.begin(),
                      map_transform(image_size, cdf_min, thrust::device_pointer_cast(d_histo.data())));

    thrust::copy(d_to_fix.begin(), d_to_fix.end(), to_fix.buffer);

    cudaDeviceSynchronize();
}
