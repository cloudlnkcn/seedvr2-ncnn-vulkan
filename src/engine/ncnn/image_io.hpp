#pragma once
#include <filesystem>
#include <mat.h>
#include <vector>

namespace seedvr2::engine::detail {
struct ImagePixels {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;
};
ImagePixels load_image(const std::filesystem::path &path);
ncnn::Mat prepare_image(const ImagePixels &input, int long_side);
void save_image(const std::filesystem::path &path, const ncnn::Mat &normalized);
} // namespace seedvr2::engine::detail
