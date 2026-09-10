// Independent cancellation oracle for long FP32 DiT projections.
#include "graph.hpp"
#include "fp32_device.hpp"
#include "linear.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        const bool host=argc==2 && std::string(argv[1])=="host";
        if (argc>2 || (argc==2 && !host)) return 2;
        try { d::init_gpu(); } catch (const std::exception &e) {std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
        if (!ncnn::get_gpu_count()) {std::cout<<"SKIP: no Vulkan device\n";return 77;}
        if(!d::fp32_device_probe(0).at("supported").get<bool>()) {
            std::cout<<"SKIP: device lacks the FP32 FMA residual prerequisite; full FP32-B inference is rejected by preflight\n";
            return 77;
        }
        constexpr int k=2560,n=8,m=4;
        ncnn::Net net;auto &opt=net.opt;
        opt.use_vulkan_compute=true;opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=false;
        opt.use_cooperative_matrix=false;opt.use_weights_in_host_memory=host;
        net.set_vulkan_device(0);
        const auto param=std::string("7767517\n2 2\nInput in0 0 1 in0\nInnerProduct linear 1 1 in0 out0 0=8 1=1 2=")+std::to_string(n*k)+"\n";
        // First word is ncnn's FP32 weight tag. Bias has no tag in InnerProduct.
        std::vector<float> model(1+n*k+n,0.f);
        for (int c=0;c<n;++c) {
            float *w=model.data()+1+c*k;const float sign=c%2?-1.f:1.f;
            for (int j=0;j<k;++j) w[j]=sign/1024.f;
            w[0]=sign*16777216.f;w[k-1]=-w[0];
            w[1]=sign*(1.f-std::ldexp(1.f,-13));w[2]=-sign;
            model[1+n*k+c]=0.f;
        }
        if (seedvr2::engine::register_dit_linear(net) || net.load_param_mem(param.c_str()) || net.load_model(reinterpret_cast<const unsigned char *>(model.data()))<=0)
            throw std::runtime_error("Cannot load projection regression");
        ncnn::Mat input(k,m,size_t(4),1);
        for (int r=0;r<m-1;++r) for (int j=0;j<k;++j) input.row(r)[j]=std::ldexp(1.f,r-1);
        for (int j=0;j<k;++j) input.row(m-1)[j]=0.f;
        input.row(m-1)[1]=1.f+std::ldexp(1.f,-13);input.row(m-1)[2]=1.f;
        const auto *device=net.vulkan_device();d::VulkanAllocators allocators(device);
        auto options=opt;options.blob_vkallocator=options.workspace_vkallocator=allocators.blob;
        options.staging_vkallocator=allocators.staging;
        ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;cmd.record_upload(input,uploaded,options);
        d::Json dispatch=d::Json::array();auto outputs=d::graph_forward<ncnn::VkMat>(net,{uploaded},&cmd,options,dispatch);
        ncnn::Mat output;cmd.record_download(outputs.at(0),output,options);
        if (cmd.submit_and_wait() || d::tensor_shape(output)!=std::vector<int>{m,n}) throw std::runtime_error("Projection execution failed");
        int failures=0;double maximum=0;
        for (int r=0;r<m;++r) for (int c=0;c<n;++c) {
            double expected=model[1+n*k+c];
            for (int j=0;j<k;++j) expected+=double(model[1+c*k+j])*input.row(r)[j];
            const float value=output.row(r)[c];const double error=std::abs(double(value)-expected);
            failures+=!std::isfinite(value)||error>(r==m-1?1e-12:1e-5);maximum=std::max(maximum,error);
        }
        std::cout<<"checked="<<m*n<<" violations="<<failures<<" max_abs="<<maximum<<'\n';
        return failures?1:0;
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
