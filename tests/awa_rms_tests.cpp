#include "awa.hpp"
#include "graph.hpp"
#include "fp32_device.hpp"
#include <array>
#include <cmath>
#include <iostream>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        if(argc!=2)return 2;const std::filesystem::path root(argv[1]);
        try{d::init_gpu();}catch(const std::exception &e){std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
        if(!ncnn::get_gpu_count())return 77;
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        std::vector<float> input(59*384),expected(59*4);
        auto read=[&](const char *name,std::vector<float>&v){std::ifstream f(root/name,std::ios::binary);f.read(reinterpret_cast<char *>(v.data()),v.size()*4);if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Invalid AWA RMS fixture");};
        read("input.f32",input);read("reference.f32",expected);
        seedvr2::engine::ExecutionTrace trace;ncnn::Net net;auto &opt=net.opt;
        opt.use_vulkan_compute=true;opt.use_packing_layout=false;opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;opt.use_bf16_storage=opt.use_bf16_packed=false;opt.use_cooperative_matrix=false;net.set_vulkan_device(0);
        constexpr char param[]="7767517\n3 4\nInput in0 0 1 in0\nInput in1 0 1 in1\nSeedVR2AWA awa 2 2 in0 in1 out0 out1 0=1 1=0 2=9.999999747e-06 3=1\n";
        alignas(16) std::array<float,512> weights;weights.fill(1.f);
        if(seedvr2::engine::register_awa(net,trace)||net.load_param_mem(param)||net.load_model(reinterpret_cast<const unsigned char *>(weights.data()))<=0)throw std::runtime_error("AWA load failed");
        std::vector<ncnn::Mat> inputs;inputs.emplace_back(384,1,1,1,size_t(4),1);inputs.emplace_back(384,58,size_t(4),1);
        std::memcpy(inputs[0].data,input.data(),384*4);std::memcpy(inputs[1].data,input.data()+384,58*384*4);
        std::size_t checked=0,failed=0;double maximum=0;
        const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);auto options=opt;
        options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;options.staging_vkallocator=allocators.staging;
        ncnn::VkCompute cmd(device);std::vector<ncnn::VkMat> uploaded(2);for(int i=0;i<2;++i)cmd.record_upload(inputs[i],uploaded[i],options);
        trace.observe_vulkan=[&](const char *name,int window,const std::vector<ncnn::VkMat> &tops,ncnn::VkCompute &command,const ncnn::Option &o){
            if(std::string(name)!="awa.qkv")return;
            if(window||tops.size()!=3||tops[0].w!=128||tops[0].h!=59||tops[0].c!=1)throw std::runtime_error("Invalid AWA trace shape");
            std::array<ncnn::Mat,2> values;for(int i=0;i<2;++i)command.record_download(tops[i],values[i],o);
            if(command.submit_and_wait()||command.reset())throw std::runtime_error("AWA trace download failed");
            for(int t=0;t<59;++t)for(int q=0;q<2;++q)for(int j=0;j<2;++j){
                float v=values[q].channel(0).row(t)[126+j],ref=expected[t*4+q*2+j];double delta=std::abs(double(v)-ref);
                ++checked;maximum=std::max(maximum,delta);failed+=!std::isfinite(v)||!std::isfinite(ref)||v!=ref;
            }
        };
        d::Json dispatch=d::Json::array();auto out=d::graph_forward<ncnn::VkMat>(net,uploaded,&cmd,options,dispatch);
        if(cmd.submit_and_wait())throw std::runtime_error("AWA submission failed");
        std::cout<<"checked="<<checked<<" max_abs="<<maximum<<" violations="<<failed<<'\n';return checked==236&&!failed?0:1;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
