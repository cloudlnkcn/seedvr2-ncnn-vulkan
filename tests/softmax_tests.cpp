#include "graph.hpp"
#include "fp32_device.hpp"
#include "softmax.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace d = seedvr2::engine::detail;
int main(int argc, char **argv) {
    try {
        if (argc != 2) return 2;
        const std::filesystem::path root(argv[1]);
        try { d::init_gpu(); }
        catch (const std::exception &e) { std::cout << "SKIP: " << e.what() << '\n'; return 77; }
        if (!ncnn::get_gpu_count()) return 77;
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        std::ifstream manifest(root / "provenance.json");
        const auto cases = d::Json::parse(manifest).at("cases");
        if (cases.size() != 12) throw std::runtime_error("Missing softmax fixtures");
        ncnn::Net net;
        auto &opt = net.opt;
        opt.use_vulkan_compute = true; opt.use_packing_layout = false;
        opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = opt.use_bf16_packed = false;
        opt.use_cooperative_matrix = false;
        net.set_vulkan_device(0);
        constexpr char param[] = "7767517\n2 2\nInput input 0 1 in0\nSoftmax softmax 1 1 in0 out0 0=-1 1=1\n";
        const unsigned char unused[4] = {};
        if (seedvr2::engine::register_fp32_softmax(net) || net.load_param_mem(param) || net.load_model(unused) != 0)
            throw std::runtime_error("Cannot load softmax graph");
        const auto *device = net.vulkan_device();
        d::VulkanAllocators allocators(device);
        auto options = opt;
        options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
        options.staging_vkallocator = allocators.staging;
        size_t checked = 0, failures = 0, changed = 0;
        double max_abs = 0, max_ulps = 0;
        for (const auto &test : cases) {
            const size_t before = failures, changed_before = changed;
            const auto shape = test.at("shape").get<std::vector<int>>();
            auto read = [&](const char *key) {
                auto tensor = d::allocate_tensor(shape);
                std::ifstream stream(root / test.at(key).get<std::string>(), std::ios::binary);
                for (int c = 0; c < tensor.c; ++c)
                    stream.read(reinterpret_cast<char *>(tensor.channel(c).data), tensor.w * tensor.h * 4);
                if (!stream || stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Invalid fixture bytes");
                return tensor;
            };
            auto input = read("input"), expected = read("reference");
            ncnn::VkCompute cmd(device);
            ncnn::VkMat uploaded;
            cmd.record_upload(input, uploaded, options);
            d::Json trace = d::Json::array();
            auto out = d::graph_forward<ncnn::VkMat>(net, {uploaded}, &cmd, options, trace);
            ncnn::Mat output;
            cmd.record_download(out.at(0), output, options);
            if (cmd.submit_and_wait() || d::tensor_shape(output) != shape)
                throw std::runtime_error("Softmax execution failed");
            for (int c = 0; c < output.c; ++c) for (int i = 0; i < output.w * output.h; ++i) {
                const float actual = output.channel(c)[i], reference = expected.channel(c)[i];
                const double delta = std::abs(double(actual) - reference);
                const double ulp = double(std::nextafter(reference, std::numeric_limits<float>::infinity())) - reference;
                max_abs = std::max(max_abs, delta); max_ulps = std::max(max_ulps, delta / ulp);
                changed += actual != reference;
                failures += !std::isfinite(actual) || !std::isfinite(reference) || delta > 2 * ulp;
                ++checked;
            }
            std::cout << "width=" << output.w << " changed=" << changed-changed_before
                      << " violations=" << failures-before << '\n';
        }
        std::cout << "checked=" << checked << " changed=" << changed << " max_abs=" << max_abs
                  << " max_ulps=" << max_ulps << " violations=" << failures << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
