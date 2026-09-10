// Isolate FP32 rotation evaluation from RMS reduction. The unrotated lanes
// expose the native normalization coefficient; the oracle rounds each multiply
// and the final addition as the original eager PyTorch operations do.
#include "awa.hpp"
#include "data/rope_fp32_table.hpp"
#include "graph.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace d = seedvr2::engine::detail;

int main(int argc, char **argv) {
    try {
        const bool gpu = argc >= 2 && std::string(argv[1]) == "vulkan";
        const bool stability = argc >= 3 && std::string(argv[2]) == "sdpa";
        const bool host = (argc == 3 && std::string(argv[2]) == "host") ||
            (argc == 4 && stability && std::string(argv[3]) == "host");
        if (argc < 2 || argc > 4 || (argc != 2 && !stability && !host) ||
            (argc == 4 && !host) || (!gpu && std::string(argv[1]) != "cpu") || (host && !gpu)) return 2;
        if (gpu) {
            try { d::init_gpu(); }
            catch (const std::exception &e) { std::cout << "SKIP: " << e.what() << '\n'; return 77; }
            if (ncnn::get_gpu_count() == 0) { std::cout << "SKIP: no Vulkan device\n"; return 77; }
        }
        double maximum = 0.;
        std::size_t checked = 0, failures = 0, nonfinite = 0;
        for (const int length : {58, 256}) {
            seedvr2::engine::ExecutionTrace trace;
            ncnn::Net net;
            auto &opt = net.opt;
            opt.use_vulkan_compute = gpu;
            opt.use_packing_layout = false;
            opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
            opt.use_bf16_storage = opt.use_bf16_packed = false;
            opt.use_cooperative_matrix = false;
            opt.use_weights_in_host_memory = host;
            opt.num_threads = 2;
            if (gpu) net.set_vulkan_device(0);
            constexpr char param[] = "7767517\n3 4\nInput in0 0 1 in0\nInput in1 0 1 in1\n"
                "SeedVR2AWA awa 2 2 in0 in1 out0 out1 0=1 1=0 2=9.999999747e-06 3=1\n";
            alignas(16) std::array<float, 512> weights;
            // Mathematically finite logits after scaling; a dot product before
            // scaling overflows. Constant V gives an independent exact oracle.
            weights.fill(1.f);
            for (int j=0;j<512;++j) if (j%128<126) weights[j]=1.f+float((j*37)%101)/32.f;
            if (seedvr2::engine::register_awa(net, trace) || net.load_param_mem(param) ||
                net.load_model(reinterpret_cast<const unsigned char *>(weights.data())) <= 0)
                throw std::runtime_error("Cannot load regression graph");
            std::vector<ncnn::Mat> inputs;
            inputs.emplace_back(384, 1, 1, 1, size_t(4), 1);
            inputs.emplace_back(384, length, size_t(4), 1);
            for (auto &input : inputs) input.fill(1.f);
            auto check = [&](const std::vector<ncnn::Mat> &qkv) {
                if (stability) return;
                if (qkv.size() != 3 || qkv[0].w != 128 || qkv[0].h != length + 1)
                    throw std::runtime_error("Unexpected gathered QKV shape");
                auto rounded=[](float a,float b) {volatile float v=a*b;return v;};
                for (int kind=0;kind<2;++kind) {
                    const auto value=qkv[kind].channel(0);
                    for (int token=0;token<=length;++token) {
                        const float scale=value.row(token)[126];
                        const int offset=(token?256:0)+kind*128;
                        for (int axis=0;axis<3;++axis) {
                            const int pos=token==0?(axis==0?length:0):token-1;
                            const float c=seedvr2::engine::rope_fp32::phase(pos,0,0),s=seedvr2::engine::rope_fp32::phase(pos,0,1);
                            for (int parity=0;parity<2;++parity) {
                                const int lane=axis*42+parity;
                                const float q=rounded(scale,weights[offset+lane]);
                                const float partner=rounded(scale,weights[offset+(lane^1)]);
                                const float expected=rounded(q,c)+(parity==0?-rounded(partner,s):rounded(partner,s));
                                const float actual=value.row(token)[lane];
                                const double error=std::abs(double(actual)-expected);
                                maximum=std::max(maximum,error);++checked;
                                nonfinite+=!std::isfinite(actual);
                                failures+=!std::isfinite(actual)||actual!=expected;
                            }
                        }
                    }
                }
            };
            auto check_output = [&](const std::vector<ncnn::Mat> &outputs) {
                if (!stability) return;
                for (const auto &output : outputs)
                    for (int channel = 0; channel < output.c; ++channel) {
                        const auto plane = output.channel(channel);
                        for (int i = 0; i < output.w * output.h * output.d; ++i) {
                            const float value = plane[i];
                            if (std::isfinite(value)) maximum = std::max(maximum, std::abs(static_cast<double>(value) - 1.));
                            else ++nonfinite;
                            ++checked;
                            failures += !std::isfinite(value) || std::abs(value - 1.f) > 1e-5f;
                        }
                    }
            };
            d::Json dispatch = d::Json::array();
            if (gpu) {
                const auto *device = net.vulkan_device();
                d::VulkanAllocators allocators(device);
                auto options = opt;
                options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
                options.staging_vkallocator = allocators.staging;
                ncnn::VkCompute command(device);
                std::vector<ncnn::VkMat> uploaded(2);
                for (std::size_t i = 0; i < inputs.size(); ++i)
                    command.record_upload(inputs[i], uploaded[i], options);
                trace.observe_vulkan = [&](const char *name, int, const std::vector<ncnn::VkMat> &tops,
                                           ncnn::VkCompute &cmd, const ncnn::Option &o) {
                    if (std::string(name) != "awa.qkv") return;
                    std::vector<ncnn::Mat> values(3);
                    for (std::size_t i = 0; i < tops.size(); ++i) cmd.record_download(tops[i], values[i], o);
                    if (cmd.submit_and_wait() || cmd.reset()) throw std::runtime_error("Download failed");
                    check(values);
                };
                auto outputs = d::graph_forward<ncnn::VkMat>(net, uploaded, &command, options, dispatch);
                std::vector<ncnn::Mat> downloaded(outputs.size());
                if (stability)
                    for (std::size_t i = 0; i < outputs.size(); ++i)
                        command.record_download(outputs[i], downloaded[i], options);
                if (command.submit_and_wait()) throw std::runtime_error("Submission failed");
                check_output(downloaded);
            } else {
                trace.observe_cpu = [&](const char *name, int, const std::vector<ncnn::Mat> &tops) {
                    if (std::string(name) == "awa.qkv") check(tops);
                };
                auto outputs = d::graph_forward<ncnn::Mat>(net, inputs, nullptr, opt, dispatch);
                check_output(outputs);
            }
            if (trace.windows != 1 || trace.sdpa_calls != 1 || trace.vulkan_calls != (gpu ? 1 : 0) ||
                trace.cpu_calls != (gpu ? 0 : 1)) throw std::runtime_error("Unexpected execution path");
        }
        std::cout << "checked=" << checked << " finite_max_abs=" << maximum
                  << " nonfinite=" << nonfinite << " violations=" << failures << '\n';
        return checked == (stability ? 40448U : 3792U) && failures == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
