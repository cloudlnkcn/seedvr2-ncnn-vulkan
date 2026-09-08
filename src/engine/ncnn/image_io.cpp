#include "image_io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 16384
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>

namespace seedvr2::engine::detail {
ImagePixels load_image(const std::filesystem::path &path) {
    constexpr std::uint64_t max_bytes = 32 * 1024 * 1024;
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > max_bytes)
        throw std::runtime_error("Image must be a PNG or JPEG file of at most 32 MiB");
    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(std::filesystem::file_size(path)));
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file)
        throw std::runtime_error("Cannot read image");
    ImagePixels image;
    int channels = 0;
    if (!stbi_info_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                               &image.width, &image.height, &channels) ||
        image.width < 16 || image.height < 16 || image.width > 16384 || image.height > 16384 ||
        std::uint64_t(image.width) * image.height > 32 * 1024 * 1024)
        throw std::runtime_error("Image dimensions must be 16..16384 with at most 32 megapixels");
    auto pixels = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>(
        stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                              &image.width, &image.height, &channels, 3), stbi_image_free);
    if (!pixels)
        throw std::runtime_error("PNG or JPEG decoding failed");
    image.rgb.assign(pixels.get(), pixels.get()+std::size_t(image.width)*image.height*3);
    return image;
}
namespace {
struct Filter { int first; std::vector<float> weights; };
float cubic(float x) {
    x = std::abs(x);
    if (x < 1.f)
        return ((1.5f*x-2.5f)*x)*x+1.f;
    if (x < 2.f)
        return ((-.5f*x+2.5f)*x-4.f)*x+2.f;
    return 0.f;
}
std::vector<Filter> filters(int source, int destination) {
    std::vector<Filter> out;
    const float scale = float(source)/float(destination);
    const float support = 2.f*std::max(scale, 1.f);
    const float inverse = 1.f/std::max(scale, 1.f);
    for (int i = 0; i < destination; ++i) {
        const float center = (float(i)+.5f)*scale;
        const int first = std::max(int(center-support+.5f), 0);
        const int end = std::min(int(center+support+.5f), source);
        Filter filter{first, {}};
        float sum = 0;
        for (int j = first; j < end; ++j) {
            const float weight = cubic((float(j)-center+.5f)*inverse);
            filter.weights.push_back(weight);
            sum += weight;
        }
        for (auto &weight : filter.weights)
            weight /= sum;
        out.push_back(std::move(filter));
    }
    return out;
}
}
ncnn::Mat prepare_image(const ImagePixels &input, int long_side) {
    if (long_side < 64 || long_side > 512 || long_side % 16)
        throw std::runtime_error("Output long side must be a multiple of 16 between 64 and 512");
    const double scale = double(long_side)/std::max(input.width, input.height);
    const int rw = int(std::nearbyint(input.width*scale));
    const int rh = int(std::nearbyint(input.height*scale));
    const int width = rw/16*16, height = rh/16*16;
    if (width < 64 || height < 64)
        throw std::runtime_error("Output short side is below 64; increase output size or crop the source");
    const int ox = int(std::nearbyint((rw-width)/2.)), oy = int(std::nearbyint((rh-height)/2.));
    auto fx = filters(input.width, rw), fy = filters(input.height, rh);
    ncnn::Mat intermediate(rw, input.height, 3, size_t(4), 1);
    ncnn::Mat output(width, height, 3, size_t(4), 1);
    if (intermediate.empty() || output.empty())
        throw std::runtime_error("Image resize allocation failed");
    for (int c = 0; c < 3; ++c) {
        auto temp = intermediate.channel(c);
        for (int y = 0; y < input.height; ++y)
            for (int x = 0; x < rw; ++x) {
                float value = 0;
                for (std::size_t j = 0; j < fx[x].weights.size(); ++j)
                    value += (input.rgb[(std::size_t(y)*input.width+fx[x].first+j)*3+c]/255.f)*fx[x].weights[j];
                temp.row(y)[x] = value;
            }
        auto dest = output.channel(c);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                float value = 0;
                const auto &filter = fy[y+oy];
                for (std::size_t j = 0; j < filter.weights.size(); ++j)
                    value += temp.row(filter.first+int(j))[x+ox]*filter.weights[j];
                dest.row(y)[x] = (std::clamp(value, 0.f, 1.f)-.5f)/.5f;
            }
    }
    return output;
}
void save_image(const std::filesystem::path &path, const ncnn::Mat &normalized) {
    if (normalized.dims != 3 || normalized.c != 3 || normalized.elempack != 1)
        throw std::runtime_error("Expected a planar RGB output tensor");
    std::vector<unsigned char> rgb(std::size_t(normalized.w)*normalized.h*3);
    for (int c = 0; c < 3; ++c) {
        auto plane = normalized.channel(c);
        for (int y = 0; y < normalized.h; ++y)
            for (int x = 0; x < normalized.w; ++x) {
                const float value = plane.row(y)[x];
                if (!std::isfinite(value))
                    throw std::runtime_error("Non-finite restored image");
                rgb[(std::size_t(y)*normalized.w+x)*3+c] = static_cast<unsigned char>(
                    std::nearbyint((std::clamp(value, -1.f, 1.f)*.5f+.5f)*255.f));
            }
    }
    if (!stbi_write_png(path.string().c_str(), normalized.w, normalized.h, 3, rgb.data(), normalized.w*3))
        throw std::runtime_error("Cannot write PNG result");
}
} // namespace seedvr2::engine::detail
