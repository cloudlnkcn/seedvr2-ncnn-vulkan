#include "video_layers.hpp"
#include "graph.hpp"
#include "fp32_device.hpp"
#include <cmath>
#include <iostream>
#include <limits>
namespace d = seedvr2::engine::detail;
int main(int argc, char **argv) {
    try {
        if (argc != 2 && argc != 3) return 2;
        const bool host = argc == 3 && std::string(argv[2]) == "host";
        if (argc == 3 && !host) return 2;
        try { d::init_gpu(); }
        catch (const std::exception &e) { std::cout << "SKIP: " << e.what() << '\n'; return 77; }
        if (!ncnn::get_gpu_count()) return 77;
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        const std::filesystem::path root(argv[1]);
        const auto fixture = d::read_document(root / "provenance.json");
        std::size_t checked = 0, failed = 0, fp64_ulp_violations = 0;
        double maximum = 0, fp64_maximum = 0;
        for (const auto &entry : fixture.at("cases")) {
            const int stride = entry.at("stride");
            const int temporal_stride = entry.value("temporal_stride", stride);
            const int kt = entry.value("temporal_kernel", 3);
            const int kernel = entry.value("kernel", 3);
            const int pad = entry.value("padding", 1), end = entry.value("end_padding", 1);
            auto input = d::allocate_tensor(entry.at("input").at("shape").get<std::vector<int>>());
            auto expected = d::allocate_tensor(entry.at("onednn_fp32").at("shape").get<std::vector<int>>());
            auto exact = d::allocate_tensor(entry.at("reference").at("shape").get<std::vector<int>>());
            d::read_tensor(d::artifact(root, entry.at("input")), input);
            d::read_tensor(d::artifact(root, entry.at("onednn_fp32")), expected);
            d::read_tensor(d::artifact(root, entry.at("reference")), exact);
            const auto weights = d::artifact(root, entry.at("weights"));
            ncnn::Net net;
            auto &opt = net.opt;
            opt.use_vulkan_compute = true; opt.use_packing_layout = false;
            opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
            opt.use_bf16_storage = opt.use_bf16_packed = false;
            opt.use_winograd_convolution = opt.use_cooperative_matrix = false;
            opt.num_threads = 4; opt.use_weights_in_host_memory = host; net.set_vulkan_device(0);
            const auto param = std::string("7767517\n2 2\nInput in0 0 1 in0\n") +
                "SeedVR2TemporalConv conv 1 1 in0 out0 0=" + std::to_string(input.c) +
                " 1=" + std::to_string(expected.c) + " 2=" + std::to_string(kt) + " 3=" + std::to_string(kernel) + " 4=" +
                std::to_string(temporal_stride) + " 5=" + std::to_string(stride) +
                " 6=" + std::to_string(pad) + " 7=" + std::to_string(end) + "\n";
            if (seedvr2::engine::register_video_layers(net) || net.load_param_mem(param.c_str()) || net.load_model(weights.c_str()))
                throw std::runtime_error("Cannot load posterior projection");
            const auto *device = net.vulkan_device();
            d::VulkanAllocators allocators(device);
            auto options = opt;
            options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
            options.staging_vkallocator = allocators.staging;
            ncnn::VkCompute cmd(device);
            ncnn::VkMat uploaded;
            cmd.record_upload(input, uploaded, options);
            d::Json trace = d::Json::array();
            auto out = d::graph_forward<ncnn::VkMat>(net, {uploaded}, &cmd, options, trace);
            ncnn::Mat output;
            cmd.record_download(out.at(0), output, options);
            if (cmd.submit_and_wait() || d::tensor_shape(output) != d::tensor_shape(expected))
                throw std::runtime_error("posterior projection shape/execution failed");
            const int spatial = expected.w * expected.h * expected.d;
            for (int c = 0; c < expected.c; ++c) for (int i = 0; i < spatial; ++i) {
                const float a = output.channel(c)[i], b = expected.channel(c)[i];
                ++checked;
                failed += !std::isfinite(a) || a != b;
                maximum = std::max(maximum, std::abs(double(a) - b));
                const float high_precision = exact.channel(c)[i];
                const double ulp = std::abs(double(std::nextafter(high_precision, std::numeric_limits<float>::infinity())) - high_precision);
                fp64_ulp_violations += !std::isfinite(a) || std::abs(double(a) - high_precision) > 2 * ulp;
                fp64_maximum = std::max(fp64_maximum, std::abs(double(a) - high_precision));
            }
        }
        if (!checked) throw std::runtime_error("No projection case was checked");
        std::cout << "checked=" << checked << " changed=" << failed << " max_abs=" << maximum << '\n';
        std::cout << "DIAGNOSTIC FP64: above_2ulp=" << fp64_ulp_violations << " max_abs=" << fp64_maximum << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
