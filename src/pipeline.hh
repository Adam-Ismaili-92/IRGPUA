#pragma once

#include <memory>
#include <regex>
#include <stack>
#include <string>

#include "image.hh"

static std::string get_number(const std::string &str) {
    std::regex r("Broken#(\\d+)");
    std::smatch match;

    if (std::regex_search(str, match, r) && match.size() > 1)
        return match.str(1);
    else {
        std::cerr << "Error file name" << std::endl;
        exit(-1);
        return "Error file name";
    }
}

struct Pipeline {
    Pipeline(const std::vector<std::string> &filepaths, bool on_gpu) {
        images = std::vector<Image>(filepaths.size());
#pragma omp parallel for
        for (std::size_t i = 0; i < filepaths.size(); ++i) {
            const int image_id = std::stoi(get_number(filepaths[i]));
            images[i] = Image(filepaths[i], on_gpu, image_id);
        }
    }

    Image &&get_image(int i) { return std::move(images[i]); }

    std::vector<Image> images;
};
