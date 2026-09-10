#include "video_layers.hpp"
#include "graph.hpp"
#include <cmath>
#include <iostream>
namespace d = seedvr2::engine::detail;
int main(int argc, char **argv) {
    try {
        if (argc != 3) return 2;
        const std::filesystem::path root(argv[1]);
        const std::string backend(argv[2]);
        if (backend != "cpu" && backend != "vulkan") return 2;
        const bool gpu = backend == "vulkan";
        if (gpu) {
            try { d::init_gpu(); }
            catch (const std::exception &e) { std::cout << "SKIP: " << e.what() << '\n'; return 77; }
            if (!ncnn::get_gpu_count()) return 77;
        }
        auto read = [&](const char *name) {
            ncnn::Mat tensor(2, 2, 3, 512, size_t(4), 1);
            std::ifstream stream(root / name, std::ios::binary);
            for (int c = 0; c < 512; ++c)
                stream.read(reinterpret_cast<char *>(tensor.channel(c).data), 12 * 4);
            if (!stream || stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Invalid fixture");
            return tensor;
        };
        std::vector<ncnn::Mat> inputs{read("q.f32"), read("k.f32"), read("v.f32")};
        auto expected = read("reference.f32");
        ncnn::Net net;
        auto &opt = net.opt;
        opt.use_vulkan_compute = gpu; opt.use_packing_layout = false;
        opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = opt.use_bf16_packed = false;
        opt.use_cooperative_matrix = false; opt.num_threads = 4;
        if (gpu) net.set_vulkan_device(0);
        constexpr char param[] = "7767517\n4 4\nInput q 0 1 in0\nInput k 0 1 in1\nInput v 0 1 in2\nSeedVR2FrameSDPA attention 3 1 in0 in1 in2 out0\n";
        const unsigned char unused[4] = {};
        if (seedvr2::engine::register_video_layers(net) || net.load_param_mem(param) || net.load_model(unused) != 0)
            throw std::runtime_error("Cannot load frame attention");
        d::Json trace = d::Json::array();
        ncnn::Mat output;
        if (gpu) {
            const auto *device = net.vulkan_device();
            d::VulkanAllocators allocators(device);
            auto options = opt;
            options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
            options.staging_vkallocator = allocators.staging;
            ncnn::VkCompute cmd(device);
            std::vector<ncnn::VkMat> uploaded(3);
            for (size_t i = 0; i < 3; ++i) cmd.record_upload(inputs[i], uploaded[i], options);
            auto out = d::graph_forward<ncnn::VkMat>(net, uploaded, &cmd, options, trace);
            cmd.record_download(out.at(0), output, options);
            if (cmd.submit_and_wait()) throw std::runtime_error("Frame attention failed");
        } else output = d::graph_forward<ncnn::Mat>(net, inputs, nullptr, opt, trace).at(0);
        if (d::tensor_shape(output) != std::vector<int>{512, 3, 2, 2}) throw std::runtime_error("Output shape differs");
        double maximum = 0;
        size_t failures = 0;
        for (int c = 0; c < 512; ++c) for (int i = 0; i < 12; ++i) {
            const float actual = output.channel(c)[i], reference = expected.channel(c)[i];
            const double delta = std::abs(double(actual) - reference);
            maximum = std::max(maximum, delta);
            failures += !std::isfinite(actual) || !std::isfinite(reference) || delta > 2e-6;
        }
        std::cout << "checked=6144 max_abs=" << maximum << " violations=" << failures << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
