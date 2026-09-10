#include "rms_norm.hpp"
#include "awa_shaders.hpp"
#include <command.h>
#include <pipeline.h>
#include <memory>
#include <cmath>
namespace seedvr2::engine {
namespace {
class DiTRMSNorm final : public ncnn::Layer {
public:
    DiTRMSNorm() {
        one_blob_only=true;support_inplace=false;support_vulkan=true;
        support_packing=support_vulkan_packing=support_vulkan_any_packing=false;
    }
    int load_param(const ncnn::ParamDict &p) override {
        return p.get(0,0)==2560 && p.get(1,0.f)==1e-5f && p.get(2,1)==0?0:-1;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if(!opt.use_vulkan_compute)return 0;
        if(!vkdev||opt.use_fp16_storage||opt.use_fp16_packed||opt.use_fp16_arithmetic||opt.use_bf16_storage||opt.use_bf16_packed)return -1;
        pipeline_=std::make_unique<ncnn::Pipeline>(vkdev);pipeline_->set_local_size_xyz(128,1,1);
        return pipeline_->create(shaders::dit_rms_norm,sizeof(shaders::dit_rms_norm),{});
    }
    int destroy_pipeline(const ncnn::Option &) override {pipeline_.reset();return 0;}
    int forward(const ncnn::Mat &x,ncnn::Mat &y,const ncnn::Option &opt) const override {
        if(x.empty()||x.n!=1||x.dims!=2||x.w!=2560||x.h<1||x.h>32768||x.elempack!=1||x.elemsize!=4)return -1;
        y.create_like(x,opt.blob_allocator);if(y.empty())return -100;
        // The locked PyTorch FP32-B oracle materializes x*x before its
        // 32-lane cascade sum. Match the shader's rounding boundaries on CPU
        // too; the upstream RMSNorm uses a different reduction and fused MACs.
        #pragma omp parallel for num_threads(opt.num_threads)
        for(int row=0;row<x.h;++row) {
            const float *input=x.row(row);float *output=y.row(row);
            float partials[32]={};
            for(int lane=0;lane<32;++lane) {
                float total=0.f;
                for(int chunk=0;chunk<5;++chunk) {
                    float acc=0.f;
                    for(int j=0;j<16;++j) {
                        const float value=input[(chunk*16+j)*32+lane];
                        volatile float squared=value*value;
                        acc=acc+squared;
                    }
                    total=total+acc;
                }
                partials[lane]=total;
            }
            float total=0.f;
            for(int d=0;d<8;++d) {
                const float s=((partials[d]+partials[8+d])+partials[16+d])+partials[24+d];
                total=total+s;
            }
            const float coefficient=1.f/std::sqrt(total/2560.f+1e-5f);
            for(int i=0;i<2560;++i)output[i]=input[i]*coefficient;
        }
        return 0;
    }
    int forward(const ncnn::VkMat &x,ncnn::VkMat &y,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        if(x.empty()||x.n!=1||x.dims!=2||x.w!=2560||x.h<1||x.h>32768||x.elempack!=1||x.elemsize!=4)return -1;
        y.create_like(x,opt.blob_vkallocator);if(y.empty())return -100;
        ncnn::Mat dispatch;dispatch.w=x.h*128;dispatch.h=dispatch.c=1;
        cmd.record_pipeline(pipeline_.get(),{x,y},{},{},dispatch);return 0;
    }
private:std::unique_ptr<ncnn::Pipeline> pipeline_;
};
ncnn::Layer *create(void *){return new DiTRMSNorm;}
}
int register_dit_rms_norm(ncnn::Net &net) {
    return net.register_custom_layer("RMSNorm",create);
}
}
