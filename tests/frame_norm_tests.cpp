// Frozen PyTorch GroupNorm arithmetic; distinct means catch cross-frame mixing.
#include "video_layers.hpp"
#include "graph.hpp"
#include "fp32_device.hpp"
#include <cmath>
#include <iostream>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        if (argc!=2) return 2;
        const std::filesystem::path root(argv[1]);
        try {d::init_gpu();} catch (const std::exception &e) {std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
        if (!ncnn::get_gpu_count()) {std::cout<<"SKIP: no Vulkan device\n";return 77;}
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        auto load=[&](const char *name,ncnn::Mat &tensor) {
            std::ifstream f(root/name,std::ios::binary);
            for (int c=0;c<tensor.c;++c) {
                auto plane=tensor.channel(c);
                f.read(reinterpret_cast<char *>(plane.data),tensor.w*tensor.h*tensor.d*4);
            }
            if (!f || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Invalid fixture length");
        };
        const auto manifest=d::read_document(root/"provenance.json");
        const auto shape=manifest.at("shape").get<std::vector<int>>();
        if (shape.size()!=4 || shape[0]<32 || shape[0]>512 || shape[0]%32 || shape[1]<1 ||
            shape[1]>17 || shape[2]<1 || shape[2]>128 || shape[3]<1 || shape[3]>128 ||
            manifest.at("groups")!=32 || manifest.at("epsilon")!=1e-6)
            throw std::runtime_error("Invalid frame normalization fixture");
        auto input=d::allocate_tensor(shape),expected=d::allocate_tensor(shape);
        ncnn::Mat affine(shape[0]*2,size_t(4));
        load("input.f32",input);load("reference.f32",expected);load("affine.f32",affine);
        ncnn::Net net;auto &opt=net.opt;
        opt.use_vulkan_compute=true;opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=false;
        opt.use_cooperative_matrix=false;net.set_vulkan_device(0);
        const auto param=std::string("7767517\n2 2\nInput in0 0 1 in0\nSeedVR2FrameNorm norm 1 1 in0 out0 0=")+std::to_string(shape[0])+" 1=32 2=1.000000e-6\n";
        if (seedvr2::engine::register_video_layers(net) || net.load_param_mem(param.c_str()) || net.load_model(reinterpret_cast<const unsigned char *>(affine.data))<=0)
            throw std::runtime_error("Cannot load frame normalization graph");
        const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
        auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;options.staging_vkallocator=allocators.staging;
        ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;cmd.record_upload(input,uploaded,options);
        d::Json dispatch=d::Json::array();auto out=d::graph_forward<ncnn::VkMat>(net,{uploaded},&cmd,options,dispatch);
        ncnn::Mat output;cmd.record_download(out.at(0),output,options);
        if (cmd.submit_and_wait() || d::tensor_shape(output)!=shape) throw std::runtime_error("Frame normalization execution failed");
        std::size_t count=0,failures=0;double maximum=0;
        const auto plane=shape[1]*shape[2]*shape[3];
        for (int c=0;c<shape[0];++c) for (int j=0;j<plane;++j) {
            const float v=output.channel(c)[j],ref=expected.channel(c)[j];
            const double delta=std::abs(double(v)-ref);maximum=std::max(maximum,delta);++count;
            failures+=!std::isfinite(v)||!std::isfinite(ref)||delta>2e-6;
        }
        std::cout<<"checked="<<count<<" max_abs="<<maximum<<" violations="<<failures<<'\n';
        return count==size_t(shape[0])*plane && !failures?0:1;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
