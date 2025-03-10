# -- Can be removed if not on OpenStack --
cmake_minimum_required(VERSION 3.18)
find_program(GCC_PATH gcc)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")
set(Thrust_DIR "/usr/local/cuda-12.2/targets/x86_64-linux/lib/cmake/thrust/")
set(CMAKE_CUDA_HOST_COMPILER ${GCC_PATH})

project(IRGPUA)

include(FindCUDAToolkit)

enable_language(CUDA)

find_package(OpenMP REQUIRED)
find_package(Thrust REQUIRED)

include_directories(src)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# set(CMAKE_CUDA_HOST_COMPILER
# /nix/store/pdlg6pavsff4aw9pil7xqg34z0z4hpk4-gcc-wrapper-9.5.0/bin/gcc)

set(COMMON_FLAGS
    "-W -Wall -Wextra -Werror -pedantic -Wno-error -Wno-pedantic -Wno-old-style-cast -fconcepts"
)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_FLAGS} -Ofast -march=native -flto -funroll-loops -ffast-math"
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COMMON_FLAGS}")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS
    "-Xcompiler -fopenmp --extended-lambda --generate-line-info --ptxas-options=-v --expt-relaxed-constexpr --default-stream per-thread ${CMAKE_CUDA_FLAGS}"
)

add_executable(main src/main.cu src/fix_cpu.cu src/fix_gpu_kk.cu src/reduce.cu
                    src/fix_gpu.cu src/fix_gpu_industrial.cu)

target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX Thrust::Thrust)

# BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
  # Default BUILD_TYPE is Release
  set(CMAKE_BUILD_TYPE Release)
  # If folder is named "build_debug" BUILD_TYPE is set to: Debug
  if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
    set(CMAKE_BUILD_TYPE Debug)
  endif()
  message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()
