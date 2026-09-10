#include "graph.hpp"
#include "video_layers.hpp"
#include <iostream>
#include <vector>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        if (argc!=2 || (std::string(argv[1])!="cpu" && std::string(argv[1])!="vulkan")) return 2;
        const bool vulkan=std::string(argv[1])=="vulkan";
        if (vulkan) {
            try { d::init_gpu(); } catch (const std::exception &e) { std::cout<<"SKIP: "<<e.what()<<'\n';return 77; }
            if (!ncnn::get_gpu_count()) { std::cout<<"SKIP: no Vulkan device\n";return 77; }
        }
        std::size_t checked=0,failed=0;double maximum=0;
        // FP32-B uses a GEMM followed by bias for 1x1x1 projections. Its
        // spatial 3x3 convolutions have a different accumulation order.
        for (const int stride : {1,2}) {
            constexpr int kernel=1;
            constexpr int ci=128,co=4,frames=5,height=3,width=4;
            constexpr float biases[co]={1.f,.5f,-.25f,0.f};
            const int ot=(frames-1)/stride+1,oh=(height-1)/stride+1,ow=(width-1)/stride+1;
            ncnn::Net net;auto &opt=net.opt;
            opt.use_vulkan_compute=vulkan;opt.use_packing_layout=false;
            opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
            opt.use_bf16_storage=opt.use_bf16_packed=false;
            opt.use_winograd_convolution=opt.use_cooperative_matrix=false;
            opt.use_sgemm_convolution=vulkan;opt.num_threads=2;
            if (vulkan) net.set_vulkan_device(0);
            const auto param=std::string("7767517\n2 2\nInput in0 0 1 in0\n")+
                "SeedVR2TemporalConv conv 1 1 in0 out0 0=128 1=4 2="+std::to_string(kernel)+
                " 3="+std::to_string(kernel)+" 4="+std::to_string(stride)+" 5="+std::to_string(stride)+" 6=0 7=0\n";
            const int count=ci*kernel*kernel*kernel;
            std::vector<float> weights(std::size_t(co)*count+co,0.f);
            for (int c=0;c<co;++c) {
                const auto offset=std::size_t(c)*count;
                const int center=(kernel/2)*kernel+kernel/2;
                weights[offset+center]=16777216.f;
                weights[offset+std::size_t((124*kernel+kernel-1)*kernel*kernel+center)]=-16777216.f;
                weights[std::size_t(co)*count+c]=biases[c];
            }
            // Input is constant one. The two spatial-center contributions cancel
            // exactly, including causal head replication; only the bias remains.
            if (seedvr2::engine::register_video_layers(net) || net.load_param_mem(param.c_str()) ||
                net.load_model(reinterpret_cast<const unsigned char *>(weights.data()))<=0)
                throw std::runtime_error("Cannot load temporal convolution regression");
            ncnn::Mat input(width,height,frames,ci,size_t(4),1);input.fill(1.f);
            ncnn::Mat output;d::Json dispatch=d::Json::array();
            if (vulkan) {
                const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
                auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;
                options.staging_vkallocator=allocators.staging;
                ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;
                cmd.record_upload(input,uploaded,options);
                auto values=d::graph_forward<ncnn::VkMat>(net,{uploaded},&cmd,options,dispatch);
                cmd.record_download(values.at(0),output,options);
                if (cmd.submit_and_wait()) throw std::runtime_error("Vulkan convolution failed");
            } else output=d::graph_forward<ncnn::Mat>(net,{input},nullptr,opt,dispatch).at(0);
            if (d::tensor_shape(output)!=std::vector<int>{co,ot,oh,ow} || dispatch.empty())
                throw std::runtime_error("Unexpected convolution output");
            for (int c=0;c<co;++c) for (int i=0;i<ot*oh*ow;++i) {
                const float actual=output.channel(c)[i];++checked;
                if (!std::isfinite(actual) || actual!=biases[c]) ++failed;
                if (std::isfinite(actual)) maximum=std::max(maximum,std::abs(double(actual)-biases[c]));
            }
        }
        std::cout<<(failed?"FAIL":"PASS")<<": pointwise temporal convolution retains bias after cancellation; checked="
                 <<checked<<" failed="<<failed<<" max_abs="<<maximum<<'\n';
        return failed?1:0;
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
