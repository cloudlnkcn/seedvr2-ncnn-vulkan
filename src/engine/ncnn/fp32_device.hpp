#pragma once
#include "graph.hpp"
#include "awa_shaders.hpp"
#include <pipeline.h>

namespace seedvr2::engine::detail {
// GLSL.std.450 Fma is not an IEEE single-rounding guarantee in core Vulkan.
// In particular Mesa llvmpipe 26.1 lowers it to multiply plus add. Test the
// arithmetic property that the FP32-B correction kernels actually need; never
// infer it from a vendor name, advertise a model certificate, or hide a failing
// model run by changing its tolerance. No global cache or CPU tensor fallback.
inline Json fp32_device_probe(int index) {
    const auto *device=ncnn::get_gpu_device(index);
    if(!device)throw std::runtime_error("Vulkan arithmetic probe: no device");
    VulkanAllocators allocators(device);
    ncnn::Option opt;
    opt.use_vulkan_compute=true;opt.use_packing_layout=false;
    opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=false;
    opt.use_bf16_storage=opt.use_bf16_packed=opt.use_cooperative_matrix=false;
    opt.blob_vkallocator=opt.workspace_vkallocator=allocators.blob;
    opt.staging_vkallocator=allocators.staging;
    ncnn::Mat input(24,size_t(4),1);
    std::array<float,8> expected{};
    for(int i=0;i<8;++i) {
        const int exponent=13+i;
        const float sign=i%2?-1.f:1.f;
        input[3*i]=std::ldexp(sign*(1.f+std::ldexp(1.f,-exponent)),i-4);
        input[3*i+1]=std::ldexp(1.f-std::ldexp(1.f,-exponent),4-i);
        input[3*i+2]=-sign;
        expected[i]=std::ldexp(-sign,-2*exponent);
    }
    ncnn::Pipeline pipeline(device);pipeline.set_local_size_xyz(8,1,1);
    if(pipeline.create(shaders::fp32_probe,sizeof(shaders::fp32_probe),{}))
        throw std::runtime_error("Cannot create Vulkan arithmetic probe");
    ncnn::VkCompute cmd(device);ncnn::VkMat uploaded;
    cmd.record_upload(input,uploaded,opt);
    if(uploaded.elempack!=1) {
        ncnn::VkMat one;device->convert_packing(uploaded,one,1,cmd,opt);uploaded=one;
    }
    ncnn::VkMat output(8,size_t(4),1,opt.blob_vkallocator);
    if(output.empty())throw std::runtime_error("Cannot allocate Vulkan arithmetic probe");
    ncnn::Mat dispatch;dispatch.w=8;dispatch.h=dispatch.c=1;
    cmd.record_pipeline(&pipeline,{uploaded,output},{},{},dispatch);
    ncnn::Mat result;cmd.record_download(output,result,opt);
    if(cmd.submit_and_wait()||result.total()!=8)
        throw std::runtime_error("Vulkan arithmetic probe execution failed");
    Json values=Json::array();unsigned failed=0;
    for(int i=0;i<8;++i) {
        const bool matched=std::isfinite(result[i])&&result[i]==expected[i];
        failed+=!matched;
        values.push_back({{"expected",expected[i]},{"actual",result[i]},{"matched",matched}});
    }
    return {{"protocol","fp32-fma-residual-v1"},{"supported",failed==0},
        {"checked",8},{"failed",failed},{"values",values},{"model_verified",false},
        {"scope","Required arithmetic prerequisite; not a full numerical or model validation"}};
}
inline Json require_fp32_device(int index) {
    auto report=fp32_device_probe(index);
    if(!report.at("supported").get<bool>())
        throw std::runtime_error("Vulkan device does not preserve FP32 fused multiply-add residuals required by SeedVR2 FP32-B; explicitly choose CPU or a device that passes the arithmetic probe in engine devices");
    return report;
}
}
