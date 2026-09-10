#include "graph.hpp"
#include "fp32_device.hpp"
#include "rms_norm.hpp"
#include <cmath>
#include <iostream>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        if(argc<2||argc>4)return 2;
        const std::filesystem::path root(argv[1]);
        const bool cpu=argc>=3 && std::string(argv[argc-1])=="--cpu";
        const bool save=argc==(cpu?4:3);
        if(!cpu) {
            try{d::init_gpu();}catch(const std::exception &e){std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
            if(!ncnn::get_gpu_count())return 77;
            if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
                std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
                return 77;
            }
        }
        ncnn::Mat input(2560,37,size_t(4),1),expected(2560,37,size_t(4),1);
        auto read=[&](const char *name,ncnn::Mat &m){std::ifstream f(root/name,std::ios::binary);f.read(reinterpret_cast<char *>(m.data),2560*37*4);if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Invalid fixture");};
        read("input.f32",input);read("reference.f32",expected);
        ncnn::Net net;auto &opt=net.opt;opt.use_vulkan_compute=!cpu;opt.num_threads=4;opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;opt.use_bf16_storage=opt.use_bf16_packed=false;opt.use_cooperative_matrix=false;if(!cpu)net.set_vulkan_device(0);
        constexpr char param[]="7767517\n2 2\nInput input 0 1 in0\nRMSNorm norm 1 1 in0 out0 0=2560 1=1.000000e-5 2=0\n";
        if(seedvr2::engine::register_dit_rms_norm(net)||net.load_param_mem(param))throw std::runtime_error("RMS graph load failed");
        const unsigned char unused[4]={};if(net.load_model(unused)!=0)throw std::runtime_error("RMS weights failed");
        ncnn::Mat output;d::Json trace=d::Json::array();
        if(cpu)output=d::graph_forward<ncnn::Mat>(net,{input},nullptr,opt,trace).at(0);
        else {
        const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
        auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;options.staging_vkallocator=allocators.staging;
        ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;cmd.record_upload(input,uploaded,options);
        auto out=d::graph_forward<ncnn::VkMat>(net,{uploaded},&cmd,options,trace);cmd.record_download(out.at(0),output,options);
        if(cmd.submit_and_wait())throw std::runtime_error("RMS execution failed");
        }
        if(d::tensor_shape(output)!=std::vector<int>{37,2560})throw std::runtime_error("RMS output shape");
        if(save){std::ofstream f(argv[2],std::ios::binary);f.write(reinterpret_cast<const char *>(output.data),2560*37*4);if(!f)throw std::runtime_error("Cannot write diagnostic tensor");}
        double maximum=0;std::size_t failures=0,nonzero=0;
        for(int i=0;i<2560*37;++i){const double delta=std::abs(double(output[i])-expected[i]);maximum=std::max(maximum,delta);nonzero+=delta!=0;failures+=!std::isfinite(output[i])||!std::isfinite(expected[i])||delta>2e-6;}
        std::cout<<"checked="<<2560*37<<" max_abs="<<maximum<<" nonzero="<<nonzero<<" violations="<<failures<<'\n';return failures?1:0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
