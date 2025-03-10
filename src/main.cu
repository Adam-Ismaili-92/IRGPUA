#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "fix_gpu_industrial.cuh"
#include "fix_gpu_kk.cuh"
#include "pipeline.hh"
#include "reduce.cuh"
using namespace std::chrono;

#define my_assert(a)                                      \
    {                                                     \
        if (!(a)) {                                       \
            throw std::runtime_error("assertion failed"); \
        }                                                 \
    }

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto &dir_entry : recursive_directory_iterator("../images")) filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline_cpu(filepaths, false);
    Pipeline pipeline_gpu(filepaths, true);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline_cpu.images.size();
    std::vector<Image> cpu_images(nb_images);
    std::vector<Image> gpu_images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i) {
        // DONE : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch
        // computations right away There are still ways to speeds this process of
        // course (wait for last class)
        cpu_images[i] = pipeline_cpu.get_image(i);
        gpu_images[i] = pipeline_gpu.get_image(i);

        auto start = high_resolution_clock::now();
        fix_image_gpu(gpu_images[i]);  // FIXME CHANGER l'algorithme utilisé, les
                                       // options: fix_image_gpu, fix_image_gpu_kk,
                                       // fix_image_gpu_industrial
        auto stop = high_resolution_clock::now();
        gpu_images[i].duration = duration_cast<milliseconds>(stop - start).count();

        start = high_resolution_clock::now();
        fix_image_cpu(cpu_images[i]);
        stop = high_resolution_clock::now();
        cpu_images[i].duration = duration_cast<milliseconds>(stop - start).count();
    }

    std::cout << "Done with compute, starting stats" << std::endl;

// -- All images are now fixed : compute stats (total then sort)

// - First compute the total of each image

// DONE : make it GPU compatible (aka faster)
// You can use multiple CPU threads for your GPU version using openmp or not
// Up to you :)
#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i) {
        auto &image = cpu_images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);

        auto &image2 = gpu_images[i];
        // image2.to_sort.total = adapt_warp_reduce(image2, image_size); //FIXME reduce thrust ou alors le nôtre ? :)
        image2.to_sort.total = thrust::reduce(thrust::host, image2.buffer, image2.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // NOT DONE OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will
    // be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, cpu_images]() mutable { return cpu_images[n++].to_sort; });

    // NOT DONE OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) { return a.total < b.total; });

    std::generate(to_sort.begin(), to_sort.end(), [n = 0, gpu_images]() mutable { return gpu_images[n++].to_sort; });

    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) { return a.total < b.total; });

    // DONE : Test here that you have the same results
    // You can compare visually and should compare image vectors values and
    // "total" values If you did the sorting, check that the ids are in the same
    // order
    for (int i = 0; i < nb_images; ++i) {
        my_assert(cpu_images[i].to_sort.id == gpu_images[i].to_sort.id);
        my_assert(cpu_images[i].to_sort.total == gpu_images[i].to_sort.total);
        const int image_size = cpu_images[i].width * cpu_images[i].height;
        for (int j = 0; j < image_size; ++j) my_assert(cpu_images[i].buffer[j] == gpu_images[i].buffer[j]);
        std::cout << "Image #" << cpu_images[i].to_sort.id << " total : " << cpu_images[i].to_sort.total << std::endl;
        std::cout << "CPU duration : " << cpu_images[i].duration << " ms | GPU duration: " << gpu_images[i].duration << " ms\n" << std::endl;
    }

    std::filesystem::create_directory("cpu");
    std::filesystem::create_directory("gpu");
    for (int i = 0; i < nb_images; ++i) {
        std::ostringstream oss;
        oss << "cpu/Image#" << cpu_images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        cpu_images[i].write(str);

        std::ostringstream oss2;
        oss2 << "gpu/Image#" << gpu_images[i].to_sort.id << ".pgm";
        std::string str2 = oss2.str();
        gpu_images[i].write(str2);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // DONE : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i) {
        if (cpu_images[i].using_cuda_malloc) {
            // std::cout << "cpu images " << i << " allocated with CUDA" << std::endl;
            cudaFreeHost(cpu_images[i].buffer);
        } else {
            // std::cout << "cpu images " << i << " allocated with MALLOC" <<
            // std::endl;
            free(cpu_images[i].buffer);
        }
    }

    for (int i = 0; i < nb_images; ++i) {
        if (gpu_images[i].using_cuda_malloc) {
            // std::cout << "gpu images " << i << " allocated with CUDA" << std::endl;
            cudaFreeHost(gpu_images[i].buffer);
        } else {
            // std::cout << "gpu images " << i << " allocated with MALLOC" <<
            // std::endl;
            free(gpu_images[i].buffer);
        }
    }

    return 0;
}
