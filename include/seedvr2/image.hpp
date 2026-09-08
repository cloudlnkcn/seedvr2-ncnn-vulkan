#pragma once
#include "seedvr2/result.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace seedvr2::engine {
struct ImageRequest {
    std::filesystem::path model_directory;
    std::filesystem::path input_file;
    std::filesystem::path output_directory;
    bool vulkan = true;
    int gpu_index = -1;
    int threads = 4;
    int long_side = 256;
    std::uint64_t seed = 666;
    bool diagnostic_tensors = false;
};
struct ImageProgress {
    std::string stage;
    int completed = 0;
    int total = 38;
    double elapsed_ms = 0;
};
using ImageObserver = std::function<void(const ImageProgress &)>;
using CancellationCheck = std::function<bool()>;

Result<std::string> inspect_image(const std::filesystem::path &path);
Result<std::string> inspect_image_package(const std::filesystem::path &directory);
Result<std::string> run_image(const ImageRequest &request, ImageObserver observer = {},
                              CancellationCheck cancelled = {});
} // namespace seedvr2::engine
