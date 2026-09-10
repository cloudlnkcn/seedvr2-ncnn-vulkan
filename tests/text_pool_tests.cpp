#include "graph.hpp"
#include "text_pool.hpp"
#include "awa_shaders.hpp"
#include <pipeline.h>
#include <cmath>
#include <iostream>
#include <limits>
namespace d=seedvr2::engine::detail;
int main(int argc,char **argv) {
    try {
        if(argc!=2)return 2;
        const std::filesystem::path root(argv[1]);
        try{d::init_gpu();}catch(const std::exception&e){std::cout<<"SKIP: "<<e.what()<<'\n';return 77;}
        if(!ncnn::get_gpu_count())return 77;
        std::ifstream manifest(root/"provenance.json");const auto cases=d::Json::parse(manifest).at("cases");
        if(cases.size()!=7)throw std::runtime_error("Missing text pooling cases");
        const auto *device=ncnn::get_gpu_device(0);d::VulkanAllocators allocators(device);
        ncnn::Option opt;opt.use_vulkan_compute=true;opt.use_packing_layout=false;
        opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
        opt.use_bf16_storage=opt.use_bf16_packed=opt.use_cooperative_matrix=false;
        opt.blob_vkallocator=opt.workspace_vkallocator=allocators.blob;opt.staging_vkallocator=allocators.staging;
        ncnn::Pipeline pipeline(device);pipeline.set_local_size_xyz(64,1,1);
        if(pipeline.create(seedvr2::engine::shaders::awa_text_mean,sizeof(seedvr2::engine::shaders::awa_text_mean),{}))throw std::runtime_error("Cannot create pooling shader");
        size_t failures=0,checked=0,changed=0;double max_abs=0;
        for(const auto &test:cases) {
            const auto shape=test.at("shape").get<std::vector<int>>();auto input=d::allocate_tensor(shape);
            ncnn::Mat expected(shape[2],shape[1],size_t(4),1);
            auto read=[&](const char *key,ncnn::Mat &x){std::ifstream f(root/test.at(key).get<std::string>(),std::ios::binary);for(int c=0;c<x.c;++c)f.read(reinterpret_cast<char*>(x.channel(c).data),x.w*x.h*4);if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Invalid pooling fixture");};
            read("input",input);read("reference",expected);
            ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;cmd.record_upload(input,uploaded,opt);
            if(uploaded.elempack!=1){ncnn::VkMat one;device->convert_packing(uploaded,one,1,cmd,opt);uploaded=one;}
            ncnn::VkMat output(expected.w,expected.h,size_t(4),1,opt.blob_vkallocator);
            auto order=seedvr2::engine::text_pool_order(test.at("lengths").get<std::vector<int>>(),shape[1]);
            std::vector<int> expected_order;for(const auto &row:test.at("order"))for(const auto &v:row)expected_order.push_back(v.get<int>());
            if(order!=expected_order)throw std::runtime_error("Text pooling metadata differs from the locked official argsort");
            ncnn::Mat order_cpu(static_cast<int>(order.size()),size_t(4),1);for(size_t i=0;i<order.size();++i)order_cpu[static_cast<int>(i)]=static_cast<float>(order[i]);
            ncnn::VkMat order_gpu;cmd.record_upload(order_cpu,order_gpu,opt);
            for(int i=0;i<expected.w*expected.h;++i){auto row=std::span<const int>(order.data()+size_t(i/expected.w)*size_t(input.c),size_t(input.c));if(seedvr2::engine::text_pool_mean(static_cast<const float*>(input.data)+i,input.cstep,row)!=expected[i])throw std::runtime_error("CPU text pooling differs");}
            std::vector<ncnn::vk_constant_type> p(4);p[0].i=expected.w*expected.h;p[1].i=input.c;p[2].i=static_cast<int>(uploaded.cstep);p[3].i=expected.w;
            ncnn::Mat dispatch;dispatch.w=p[0].i;dispatch.h=dispatch.c=1;
            cmd.record_pipeline(&pipeline,{uploaded,output,order_gpu},{},p,dispatch);
            ncnn::Mat result;cmd.record_download(output,result,opt);
            if(cmd.submit_and_wait()||d::tensor_shape(result)!=std::vector<int>{shape[1],shape[2]})throw std::runtime_error("Pooling execution failed");
            size_t before=failures;
            for(int i=0;i<expected.w*expected.h;++i){const float a=result[i],b=expected[i];double delta=std::abs(double(a)-b);max_abs=std::max(max_abs,delta);failures+=!std::isfinite(a)||!std::isfinite(b)||a!=b;changed+=a!=b;++checked;}
            std::cout<<"windows="<<input.c<<" violations="<<failures-before<<'\n';
        }
        std::cout<<"checked="<<checked<<" changed="<<changed<<" max_abs="<<max_abs<<" violations="<<failures<<'\n';return failures?1:0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
