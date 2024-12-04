#pragma once

#include "image.hh"

int crappy_reduce_gpu(Image image, int image_size);

int adapt_warp_reduce(Image image, int image_size);

int register_warp_reduce(Image image, int image_size);
