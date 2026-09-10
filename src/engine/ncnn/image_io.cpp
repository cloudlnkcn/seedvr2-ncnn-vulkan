#include "image_io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_PNG
#define STBI_MAX_DIMENSIONS 16384
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>

namespace seedvr2::engine::detail {
namespace {
// libjpeg uses longjmp for errors. Keep mutable C decoder state on the heap;
// do not cross C++ object construction/destruction inside the protected region.
struct JpegState {
    jpeg_decompress_struct decoder{};
    jpeg_error_mgr error{};
    std::jmp_buf jump{};
    unsigned char *pixels = nullptr;
    bool created = false, warned = false;
    char message[JMSG_LENGTH_MAX]{};
    ~JpegState() {
        if (created) jpeg_destroy_decompress(&decoder);
        std::free(pixels);
    }
};
void jpeg_failure(j_common_ptr info) {
    auto *state = static_cast<JpegState *>(info->client_data);
    info->err->format_message(info, state->message);
    std::longjmp(state->jump, 1);
}
void jpeg_message(j_common_ptr info, int level) {
    if (level < 0) static_cast<JpegState *>(info->client_data)->warned = true;
}
ImagePixels load_jpeg(const std::vector<unsigned char> &bytes) {
    auto state = std::make_unique<JpegState>();
    auto &d = state->decoder;
    d.err = jpeg_std_error(&state->error);
    state->error.error_exit = jpeg_failure;
    state->error.emit_message = jpeg_message;
    d.client_data = state.get();
    if (setjmp(state->jump)) throw std::runtime_error(std::string("JPEG decoding failed: ")+state->message);
    state->created = true;
    jpeg_create_decompress(&d);
    jpeg_mem_src(&d, bytes.data(), static_cast<unsigned long>(bytes.size()));
    jpeg_read_header(&d, TRUE);
    if (d.image_width < 16 || d.image_height < 16 || d.image_width > 16384 || d.image_height > 16384 ||
        std::uint64_t(d.image_width)*d.image_height > 32*1024*1024)
        throw std::runtime_error("Image dimensions must be 16..16384 with at most 32 megapixels");
    if (d.jpeg_color_space != JCS_YCbCr && d.jpeg_color_space != JCS_RGB && d.jpeg_color_space != JCS_GRAYSCALE)
        throw std::runtime_error("JPEG must use RGB, YCbCr or grayscale; convert CMYK to RGB first");
    d.out_color_space = JCS_RGB;
    d.dct_method = JDCT_ISLOW;
    d.do_fancy_upsampling = TRUE;
    d.mem->max_memory_to_use = 64*1024*1024;
    jpeg_start_decompress(&d);
    const std::size_t stride = std::size_t(d.output_width)*3;
    state->pixels = static_cast<unsigned char *>(std::malloc(stride*d.output_height));
    if (!state->pixels) throw std::runtime_error("JPEG pixel allocation failed");
    while (d.output_scanline < d.output_height) {
        JSAMPROW row = state->pixels+std::size_t(d.output_scanline)*stride;
        if (jpeg_read_scanlines(&d, &row, 1) != 1) throw std::runtime_error("Incomplete JPEG scanline");
    }
    jpeg_finish_decompress(&d);
    if (state->warned) throw std::runtime_error("JPEG is truncated or contains invalid data");
    ImagePixels image{static_cast<int>(d.output_width), static_cast<int>(d.output_height), {}};
    image.rgb.assign(state->pixels, state->pixels+stride*d.output_height);
    return image;
}
}
ImagePixels load_image(const std::filesystem::path &path) {
    constexpr std::uint64_t max_bytes = 32 * 1024 * 1024;
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > max_bytes)
        throw std::runtime_error("Image must be a PNG or JPEG file of at most 32 MiB");
    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(std::filesystem::file_size(path)));
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file)
        throw std::runtime_error("Cannot read image");
    if (bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8)
        return load_jpeg(bytes);
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
        return std::fma(std::fma(1.5f,x,-2.5f)*x,x,1.f);
    if (x < 2.f)
        return std::fma(std::fma(std::fma(-.5f,x,2.5f),x,-4.f),x,2.f);
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
float rounded_product(float a,float b) {
    // Keep the SIMD-batch products rounded separately, including on compilers
    // that contract a multiplication with its subsequent addition by default.
    volatile float product=a*b;
    return product;
}
template<class Read> float resample(const Filter &filter,Read read) {
    // Locked FP32-B antialias evaluation: first term, four-term batches with
    // separately rounded products, then fused scalar tail. Coefficients use
    // the same fused cubic polynomial. Applying FMA to every term is different.
    const auto size=filter.weights.size();
    float value=rounded_product(read(0),filter.weights[0]);
    const auto end=1+(size-1)/4*4;
    for (std::size_t j=1;j<end;++j) value+=rounded_product(read(j),filter.weights[j]);
    for (std::size_t j=end;j<size;++j) value=std::fma(read(j),filter.weights[j],value);
    return value;
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
                temp.row(y)[x] = resample(fx[x],[&](std::size_t j) {
                    return input.rgb[(std::size_t(y)*input.width+fx[x].first+j)*3+c]/255.f;
                });
            }
        auto dest = output.channel(c);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto &filter = fy[y+oy];
                const float value = resample(filter,[&](std::size_t j) {return temp.row(filter.first+int(j))[x+ox];});
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
