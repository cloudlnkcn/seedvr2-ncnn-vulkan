#include "linear.hpp"
#include "awa_shaders.hpp"
#include <array>
#include <command.h>
#include <memory>
#include <pipeline.h>

namespace seedvr2::engine {
namespace {
class Linear final : public ncnn::Layer {
public:
    Linear() {
        one_blob_only=true;support_inplace=false;support_vulkan=true;
        support_packing=support_vulkan_packing=support_vulkan_any_packing=false;
    }
    int load_param(const ncnn::ParamDict &p) override {
        out_=p.get(0,0);bias_=p.get(1,0);count_=p.get(2,0);
        if (out_<=0 || out_>7680 || out_%4 || count_<=0 || count_%out_ ||
            (bias_!=0 && bias_!=1) || p.get(8,0)!=0 || p.get(9,0)!=0) return -1;
        in_=count_/out_;
        return in_>0 && in_<=6912?0:-1;
    }
    int load_model(const ncnn::ModelBin &mb) override {
        const auto weights=mb.load(count_,0);
        const auto bias=bias_?mb.load(out_,1):ncnn::Mat();
        if (weights.empty() || (bias_ && bias.empty()) || weights.elemsize!=4 || (bias_ && bias.elemsize!=4)) return -100;
        packed_.create(in_,out_/4,size_t(16),4);
        bias_data_.create(out_,size_t(4));
        if (packed_.empty() || bias_data_.empty()) return -100;
        bias_data_.fill(0.f);
        for (int c=0;c<out_;c+=4) {
            float *dst=packed_.row(c/4);
            for (int j=0;j<in_;++j) for (int lane=0;lane<4;++lane)
                dst[4*j+lane]=weights[(c+lane)*in_+j];
        }
        if (bias_) for (int c=0;c<out_;++c) bias_data_[c]=bias[c];
        return 0;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (!opt.use_vulkan_compute || !vkdev || opt.use_fp16_storage || opt.use_fp16_packed ||
            opt.use_fp16_arithmetic || opt.use_bf16_storage || opt.use_bf16_packed) return -1;
        pipeline_=std::make_unique<ncnn::Pipeline>(vkdev);
        pipeline_->set_local_size_xyz(64,1,1);
        return pipeline_->create(shaders::dit_linear,sizeof(shaders::dit_linear),{});
    }
    int upload_model(ncnn::VkTransfer &cmd,const ncnn::Option &opt) override {
        cmd.record_upload(packed_,weights_gpu_,opt,false);
        cmd.record_upload(bias_data_,bias_gpu_,opt,false);
        if (weights_gpu_.empty() || bias_gpu_.empty()) return -100;
        if (opt.lightmode) {packed_.release();bias_data_.release();}
        return 0;
    }
    int destroy_pipeline(const ncnn::Option &) override {
        pipeline_.reset();weights_gpu_.release();bias_gpu_.release();return 0;
    }
    int forward(const ncnn::VkMat &x,ncnn::VkMat &y,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        if (x.empty() || x.n!=1 || x.dims!=2 || x.w!=in_ || x.h<1 || x.h>1024 ||
            x.elempack!=1 || x.elemsize!=4 || !pipeline_ || weights_gpu_.empty() || bias_gpu_.empty()) return -1;
        y.create(out_,x.h,size_t(4),1,opt.blob_vkallocator);
        if (y.empty()) return -100;
        std::vector<ncnn::vk_constant_type> p(3);p[0].i=in_;p[1].i=out_;p[2].i=x.h;
        ncnn::Mat dispatch;dispatch.w=out_/4;dispatch.h=x.h;dispatch.c=1;
        cmd.record_pipeline(pipeline_.get(),{x,weights_gpu_,bias_gpu_,y},{},p,dispatch);
        return 0;
    }
private:
    int in_=0,out_=0,count_=0,bias_=0;
    ncnn::Mat packed_,bias_data_;
    ncnn::VkMat weights_gpu_,bias_gpu_;
    std::unique_ptr<ncnn::Pipeline> pipeline_;
};
ncnn::Layer *create(void *) {return new Linear;}
}
int register_dit_linear(ncnn::Net &net) {
    return net.opt.use_vulkan_compute?net.register_custom_layer("InnerProduct",create):0;
}
}
