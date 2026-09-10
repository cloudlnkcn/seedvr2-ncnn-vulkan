#include "graph.hpp"
#include "fp32_device.hpp"
#include "silu.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace d = seedvr2::engine::detail;

int main(int argc, char **argv) {
    try {
        if (argc < 2 || argc > 3) return 2;
        const std::filesystem::path root(argv[1]);
        try { d::init_gpu(); }
        catch (const std::exception &e) { std::cout << "SKIP: " << e.what() << '\n'; return 77; }
        if (!ncnn::get_gpu_count()) return 77;
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        const auto count = std::filesystem::file_size(root / "input.f32") / 4;
        if (!count || count > 2'000'000) throw std::runtime_error("Invalid fixture length");
        ncnn::Mat input(static_cast<int>(count), size_t(4), 1), expected;
        expected.create_like(input);
        auto read = [&](const char *name, ncnn::Mat &tensor) {
            std::ifstream stream(root / name, std::ios::binary);
            stream.read(reinterpret_cast<char *>(tensor.data), count * 4);
            if (!stream || stream.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Invalid fixture bytes");
        };
        read("input.f32", input); read("reference.f32", expected);
        ncnn::Net net;
        auto &opt = net.opt;
        opt.use_vulkan_compute = true; opt.use_packing_layout = false;
        opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = opt.use_bf16_packed = false;
        opt.use_cooperative_matrix = false;
        net.set_vulkan_device(0);
        constexpr char param[] = "7767517\n2 2\nInput input 0 1 in0\nSwish silu 1 1 in0 out0\n";
        if (seedvr2::engine::register_dit_silu(net) || net.load_param_mem(param))
            throw std::runtime_error("SiLU graph load failed");
        const unsigned char unused[4] = {};
        if (net.load_model(unused) != 0) throw std::runtime_error("SiLU pipeline failed");
        const auto *device = net.vulkan_device();
        d::VulkanAllocators allocators(device);
        auto options = opt;
        options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
        options.staging_vkallocator = allocators.staging;
        double maximum = 0, max_ulps = 0;
        std::size_t failures = 0, checked = 0;
        const std::vector<std::vector<int>> shapes{{static_cast<int>(count)}, {1322, 31}, {53, 7, 17, 13}};
        for (const auto &shape : shapes) {
            auto tensor = shape.size() == 1 ? ncnn::Mat(shape[0], size_t(4), 1) : d::allocate_tensor(shape);
            const auto plane = size_t(tensor.w) * tensor.h * tensor.d;
            const auto logical_count = plane * tensor.c;
            for (size_t i = 0; i < logical_count; ++i)
                reinterpret_cast<float *>(tensor.data)[(i / plane) * tensor.cstep + i % plane] = input[i % count];
            ncnn::VkCompute cmd(device);
            ncnn::VkMat uploaded;
            cmd.record_upload(tensor, uploaded, options);
            d::Json trace = d::Json::array();
            auto out = d::graph_forward<ncnn::VkMat>(net, {uploaded}, &cmd, options, trace);
            ncnn::Mat output;
            cmd.record_download(out.at(0), output, options);
            if (cmd.submit_and_wait() || (shape.size() == 1 ? output.dims != 1 || output.w != shape[0]
                                                                         : d::tensor_shape(output) != shape))
                throw std::runtime_error("SiLU execution failed");
            if (argc == 3 && shape.size() == 1) {
                std::ofstream stream(argv[2], std::ios::binary);
                stream.write(reinterpret_cast<const char *>(output.data), count * 4);
                if (!stream) throw std::runtime_error("Cannot write diagnostic tensor");
            }
            for (std::size_t i = 0; i < logical_count; ++i) {
                const float actual = reinterpret_cast<const float *>(output.data)[(i / plane) * output.cstep + i % plane];
                const float reference = expected[i % count];
                const double delta = std::abs(double(actual) - reference);
                const float magnitude = std::abs(reference);
                const double ulp = double(std::nextafter(magnitude, std::numeric_limits<float>::infinity())) - magnitude;
                maximum = std::max(maximum, delta);
                max_ulps = std::max(max_ulps, delta / ulp);
                // Two ULPs allow the independent CPU exponential's rounding error.
                failures += !std::isfinite(actual) || !std::isfinite(reference) || delta > 2 * ulp;
            }
            checked += logical_count;
        }
        std::cout << "checked=" << checked << " shapes=" << shapes.size()
                  << " max_abs=" << maximum << " max_ulps=" << max_ulps
                  << " violations=" << failures << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
