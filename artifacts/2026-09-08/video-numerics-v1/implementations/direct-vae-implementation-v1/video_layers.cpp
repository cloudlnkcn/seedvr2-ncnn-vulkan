#include "video_layers.hpp"
#include "awa_shaders.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <command.h>
#include <cstring>
#include <memory>
#include <modelbin.h>
#include <pipeline.h>

namespace seedvr2::engine {
namespace {
constexpr std::uint64_t max_elements = 128ULL * 1024 * 1024;
constexpr std::uint64_t max_gather_elements = 256ULL * 1024 * 1024;
template<class M> bool valid(const M &m) {
    return !m.empty() && m.n == 1 && m.dims == 4 && m.elemsize == 4 && m.elempack == 1 &&
        m.w > 0 && m.w <= 256 && m.h > 0 && m.h <= 256 && m.d > 0 && m.d <= 17 &&
        m.c > 0 && m.c <= 4096 && std::uint64_t(m.c)*m.d*m.h*m.w <= max_elements;
}
bool finite(const ncnn::Mat &x) {
    if (x.empty()) return false;
    const auto *p = static_cast<const float *>(x);
    for (std::size_t i=0;i<x.total();++i) if (!std::isfinite(p[i])) return false;
    return true;
}
class VideoLayer : public ncnn::Layer {
public:
    VideoLayer() {
        one_blob_only=true; support_inplace=false; support_vulkan=true;
        support_packing=support_vulkan_packing=support_vulkan_any_packing=false;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_fp16_arithmetic || opt.use_bf16_storage || opt.use_bf16_packed) return -1;
        if (!opt.use_vulkan_compute) return 0;
        if (!vkdev) return -1;
        map_=std::make_unique<ncnn::Pipeline>(vkdev);
        map_->set_local_size_xyz(128,1,1);
        return map_->create(shaders::video_map,sizeof(shaders::video_map),{});
    }
    int destroy_pipeline(const ncnn::Option &) override { map_.reset(); return 0; }
protected:
    void map(const ncnn::VkMat &x, const ncnn::VkMat &y, std::array<int,19> p,
             ncnn::VkCompute &cmd) const {
        std::vector<ncnn::vk_constant_type> constants(p.size());
        for (std::size_t i=0;i<p.size();++i) constants[i].i=p[i];
        ncnn::Mat dispatch;
        dispatch.w=std::min(p[1],65536); dispatch.h=(p[1]+65535)/65536; dispatch.c=1;
        cmd.record_pipeline(map_.get(),{x,y},{},constants,dispatch);
    }
    std::unique_ptr<ncnn::Pipeline> map_;
};

class TemporalConv final : public VideoLayer {
public:
    int load_param(const ncnn::ParamDict &p) override {
        ci_=p.get(0,0); co_=p.get(1,0); kt_=p.get(2,0); k_=p.get(3,0);
        st_=p.get(4,0); s_=p.get(5,0); pad_=p.get(6,-1); end_=p.get(7,-1);
        return ci_>0 && ci_<=4096 && co_>0 && co_<=4096 && (kt_==1||kt_==3) &&
            (k_==1||k_==3) && (st_==1||st_==2) && (s_==1||s_==2) &&
            (pad_==0||pad_==1) && (end_==0||end_==1) &&
            std::uint64_t(ci_)*co_*kt_*k_*k_<=max_elements ? 0:-1;
    }
    int load_model(const ncnn::ModelBin &mb) override {
        weights_[0]=mb.load(ci_*co_*kt_*k_*k_,1); weights_[1]=mb.load(co_,1);
        return finite(weights_[0]) && finite(weights_[1]) ? 0:-1;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (VideoLayer::create_pipeline(opt)!=0) return -1;
        convolution_.reset(opt.use_vulkan_compute ? ncnn::create_layer_vulkan("Convolution") : ncnn::create_layer_cpu("Convolution"));
        if (!convolution_) return -1;
        convolution_->vkdev=vkdev;
        ncnn::ParamDict p;
        p.set(0,co_);p.set(1,k_);p.set(3,s_);p.set(4,0);p.set(5,1);p.set(6,ci_*co_*kt_*k_*k_);
        ncnn::ModelBinFromMatArray mb(weights_.data());
        return convolution_->load_param(p)==0 && convolution_->load_model(mb)==0 &&
            convolution_->create_pipeline(convolution_options(opt))==0 ? 0:-1;
    }
    int upload_model(ncnn::VkTransfer &cmd,const ncnn::Option &opt) override {
        return convolution_->upload_model(cmd,opt);
    }
    int destroy_pipeline(const ncnn::Option &opt) override {
        if (convolution_) convolution_->destroy_pipeline(convolution_options(opt));
        convolution_.reset(); return VideoLayer::destroy_pipeline(opt);
    }
    template<class M> bool geometry(const M &x, int &t,int &h,int &w,int &tile) const {
        if (!valid(x) || x.c!=ci_) return false;
        t=(x.d-1)/st_+1;h=(x.h+pad_+end_-k_)/s_+1;w=(x.w+pad_+end_-k_)/s_+1;
        tile=((x.h+pad_+end_+s_-1)/s_)*s_;
        return h>0 && w>0 && std::uint64_t(ci_)*kt_*t*tile*(x.w+pad_+end_)<=max_gather_elements &&
            std::uint64_t(co_)*t*h*w<=max_elements;
    }
    int forward(const ncnn::Mat &x,ncnn::Mat &y,const ncnn::Option &opt) const override {
        int t,h,w,tile;
        if (!geometry(x,t,h,w,tile)) return -1;
        ncnn::Mat gather(x.w+pad_+end_,t*tile,ci_*kt_,size_t(4),1,opt.workspace_allocator);
        if (gather.empty()) return -100;
        gather.fill(0.f);
        for (int c=0;c<ci_*kt_;++c) {
            auto *out=static_cast<float *>(gather.channel(c));
            const auto *in=static_cast<const float *>(x.channel(c/kt_));
            for (int f=0;f<t;++f) {
                const int tf=std::clamp(f*st_-(kt_-1)+c%kt_,0,x.d-1);
                for (int row=0;row<x.h;++row)
                    std::memcpy(out+(f*tile+row+pad_)*gather.w+pad_,in+(tf*x.h+row)*x.w,std::size_t(x.w)*4);
            }
        }
        ncnn::Mat result;
        if (convolution_->forward(gather,result,convolution_options(opt))!=0 || result.empty() || result.elempack!=1) return -1;
        y.create(w,h,t,co_,size_t(4),1,opt.blob_allocator);
        if (y.empty()) return -100;
        for (int c=0;c<co_;++c) for (int f=0;f<t;++f) for (int row=0;row<h;++row)
            std::memcpy(static_cast<float *>(y.channel(c))+(f*h+row)*w,
                static_cast<const float *>(result.channel(c))+(f*(tile/s_)+row)*result.w,std::size_t(w)*4);
        return 0;
    }
    int forward(const ncnn::VkMat &x,ncnn::VkMat &y,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        int t,h,w,tile;
        if (!geometry(x,t,h,w,tile)) return -1;
        ncnn::VkMat gather(x.w+pad_+end_,t*tile,ci_*kt_,size_t(4),1,opt.workspace_vkallocator);
        if (gather.empty()) return -100;
        map(x,gather,{0,gather.w*gather.h*ci_*kt_,x.w,x.h,x.d,x.c,int(x.cstep),
            gather.w,gather.h,1,ci_*kt_,int(gather.cstep),kt_,st_,s_,pad_,tile,0,0},cmd);
        const int pack=(ci_*kt_)%4==0?4:1;
        ncnn::VkMat packed=gather;
        if (pack!=1) vkdev->convert_packing(gather,packed,pack,cmd,opt);
        ncnn::VkMat result;
        if (packed.empty() || convolution_->forward(packed,result,cmd,opt)!=0 || result.empty()) return -1;
        if (result.elempack!=1) {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(result,unpacked,1,cmd,opt); result=unpacked;
        }
        y.create(w,h,t,co_,size_t(4),1,opt.blob_vkallocator);
        if (y.empty() || result.empty()) return -100;
        map(result,y,{1,w*h*t*co_,result.w,result.h,1,co_,int(result.cstep),w,h,t,co_,int(y.cstep),kt_,st_,s_,pad_,tile/s_,0,0},cmd);
        return 0;
    }
private:
    static ncnn::Option convolution_options(const ncnn::Option &opt) {
        auto result=opt;
        // In the retained 17-frame FP32-B encoder, direct CPU convolution
        // reduces posterior error versus SGEMM before the 32-block trajectory
        // amplifies it. Keep this accuracy choice local to temporal VAE layers.
        if (!opt.use_vulkan_compute) result.use_sgemm_convolution=false;
        return result;
    }
    int ci_=0,co_=0,kt_=0,k_=0,st_=0,s_=0,pad_=0,end_=0;
    std::array<ncnn::Mat,2> weights_;
    std::unique_ptr<ncnn::Layer> convolution_;
};

class FrameNorm final : public VideoLayer {
public:
    int load_param(const ncnn::ParamDict &p) override {
        channels_=p.get(0,0); groups_=p.get(1,0); epsilon_=p.get(2,0.f);
        return channels_>0 && channels_<=512 && groups_==32 && channels_%groups_==0 && epsilon_==1e-6f ? 0:-1;
    }
    int load_model(const ncnn::ModelBin &mb) override {
        const auto w=mb.load(channels_,1), b=mb.load(channels_,1);
        if (!finite(w)||!finite(b)) return -1;
        affine_.create(2*channels_,size_t(4));
        if (affine_.empty()) return -100;
        std::memcpy(affine_.data,w.data,std::size_t(channels_)*4);
        std::memcpy(static_cast<float *>(affine_)+channels_,b.data,std::size_t(channels_)*4);
        return 0;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (VideoLayer::create_pipeline(opt)!=0) return -1;
        if (!opt.use_vulkan_compute) return 0;
        norm_=std::make_unique<ncnn::Pipeline>(vkdev);norm_->set_local_size_xyz(128,1,1);
        return norm_->create(shaders::video_norm,sizeof(shaders::video_norm),{});
    }
    int upload_model(ncnn::VkTransfer &cmd,const ncnn::Option &opt) override {
        cmd.record_upload(affine_,gpu_,opt);return gpu_.empty()?-100:0;
    }
    int destroy_pipeline(const ncnn::Option &opt) override {
        norm_.reset();gpu_.release();return VideoLayer::destroy_pipeline(opt);
    }
    int forward(const ncnn::Mat &x,ncnn::Mat &y,const ncnn::Option &opt) const override {
        if (!valid(x)||x.c!=channels_) return -1;
        y.create_like(x,opt.blob_allocator);if (y.empty()) return -100;
        const int spatial=x.w*x.h, cg=channels_/groups_, count=cg*spatial;
        for (int t=0;t<x.d;++t) for (int g=0;g<groups_;++g) {
            double sum=0,variance=0;
            for (int c=g*cg;c<(g+1)*cg;++c) {
                const auto *in=static_cast<const float *>(x.channel(c))+t*spatial;
                for (int i=0;i<spatial;++i) sum+=in[i];
            }
            const float mean=static_cast<float>(sum/count);
            for (int c=g*cg;c<(g+1)*cg;++c) {
                const auto *in=static_cast<const float *>(x.channel(c))+t*spatial;
                for (int i=0;i<spatial;++i) { double v=double(in[i])-mean;variance+=v*v; }
            }
            const float scale=1.f/std::sqrt(static_cast<float>(variance/count)+epsilon_);
            for (int c=g*cg;c<(g+1)*cg;++c) {
                const auto *in=static_cast<const float *>(x.channel(c))+t*spatial;
                auto *out=static_cast<float *>(y.channel(c))+t*spatial;
                for (int i=0;i<spatial;++i) out[i]=(in[i]-mean)*scale*affine_[c]+affine_[channels_+c];
            }
        }
        return 0;
    }
    int forward(const ncnn::VkMat &x,ncnn::VkMat &y,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        if (!valid(x)||x.c!=channels_) return -1;
        y.create_like(x,opt.blob_vkallocator);if (y.empty()) return -100;
        std::vector<ncnn::vk_constant_type> p(7);
        const std::array<int,6> values={x.w*x.h,x.d,channels_,int(x.cstep),int(y.cstep),groups_};
        for (std::size_t i=0;i<values.size();++i) p[i].i=values[i];
        p[6].f=epsilon_;
        ncnn::Mat dispatch;dispatch.w=groups_*128;dispatch.h=x.d;dispatch.c=1;
        cmd.record_pipeline(norm_.get(),{x,gpu_,y},{},p,dispatch);return 0;
    }
private:
    int channels_=0,groups_=0;float epsilon_=0;
    ncnn::Mat affine_;ncnn::VkMat gpu_;std::unique_ptr<ncnn::Pipeline> norm_;
};

class TemporalShuffle final : public VideoLayer {
public:
    int load_param(const ncnn::ParamDict &p) override {
        channels_=p.get(0,0);ratio_=p.get(1,0);
        return channels_>0 && channels_<=512 && (ratio_==1||ratio_==2)?0:-1;
    }
    template<class M> bool prepare(const M &x) const {
        return valid(x) && x.c==4*ratio_*channels_ && x.w<=128 && x.h<=128 && (x.d-1)*ratio_+1<=17;
    }
    int forward(const ncnn::Mat &x,ncnn::Mat &y,const ncnn::Option &opt) const override {
        if (!prepare(x)) return -1;
        y.create(x.w*2,x.h*2,(x.d-1)*ratio_+1,channels_,size_t(4),1,opt.blob_allocator);
        if (y.empty()) return -100;
        for (int c=0;c<channels_;++c) for (int t=0;t<y.d;++t) for (int h=0;h<y.h;++h) for (int w=0;w<y.w;++w) {
            const int expanded=t==0?0:t+ratio_-1;
            const int sc=(((h%2)*2+w%2)*ratio_+expanded%ratio_)*channels_+c;
            static_cast<float *>(y.channel(c))[(t*y.h+h)*y.w+w]=static_cast<const float *>(x.channel(sc))[((expanded/ratio_)*x.h+h/2)*x.w+w/2];
        }
        return 0;
    }
    int forward(const ncnn::VkMat &x,ncnn::VkMat &y,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        if (!prepare(x)) return -1;
        y.create(x.w*2,x.h*2,(x.d-1)*ratio_+1,channels_,size_t(4),1,opt.blob_vkallocator);
        if (y.empty()) return -100;
        map(x,y,{4,y.w*y.h*y.d*y.c,x.w,x.h,x.d,x.c,int(x.cstep),y.w,y.h,y.d,y.c,int(y.cstep),0,0,0,0,0,0,ratio_},cmd);
        return 0;
    }
private:int channels_=0,ratio_=0;
};

class FrameSDPA final : public VideoLayer {
public:
    FrameSDPA() {one_blob_only=false;}
    int create_pipeline(const ncnn::Option &opt) override {
        if (VideoLayer::create_pipeline(opt)!=0) return -1;
        sdpa_.reset(opt.use_vulkan_compute?ncnn::create_layer_vulkan("SDPA"):ncnn::create_layer_cpu("SDPA"));
        if (!sdpa_) return -1;
        sdpa_->vkdev=vkdev;ncnn::ParamDict p;
        return sdpa_->load_param(p)==0 && sdpa_->create_pipeline(opt)==0 ? 0:-1;
    }
    int destroy_pipeline(const ncnn::Option &opt) override {
        if (sdpa_) sdpa_->destroy_pipeline(opt);
        sdpa_.reset();return VideoLayer::destroy_pipeline(opt);
    }
    template<class M> bool prepare(const std::vector<M> &x) const {
        if (x.size()!=3) return false;
        for (const auto &v:x) if (!valid(v)||v.c!=512||v.w*v.h>1024 || v.w!=x[0].w || v.h!=x[0].h || v.d!=x[0].d) return false;
        return true;
    }
    int forward(const std::vector<ncnn::Mat> &x,std::vector<ncnn::Mat> &out,const ncnn::Option &opt) const override {
        if (!prepare(x)||out.size()!=1) return -1;
        auto &y=out[0];y.create_like(x[0],opt.blob_allocator);if (y.empty()) return -100;
        const int n=y.w*y.h;
        for (int t=0;t<y.d;++t) {
            std::vector<ncnn::Mat> qkv(3), result(1);
            for (int a=0;a<3;++a) {
                qkv[a].create(y.c,n,1,size_t(4),1,opt.workspace_allocator);if (qkv[a].empty()) return -100;
                auto *dst=static_cast<float *>(qkv[a]);
                for (int c=0;c<y.c;++c) for (int i=0;i<n;++i) dst[i*y.c+c]=static_cast<const float *>(x[a].channel(c))[t*n+i];
            }
            if (sdpa_->forward(qkv,result,opt)!=0||result[0].empty()) return -1;
            const auto *src=static_cast<const float *>(result[0]);
            for (int c=0;c<y.c;++c) for (int i=0;i<n;++i) static_cast<float *>(y.channel(c))[t*n+i]=src[i*y.c+c];
        }
        return 0;
    }
    int forward(const std::vector<ncnn::VkMat> &x,std::vector<ncnn::VkMat> &out,ncnn::VkCompute &cmd,const ncnn::Option &opt) const override {
        if (!prepare(x)||out.size()!=1) return -1;
        auto &y=out[0];y.create_like(x[0],opt.blob_vkallocator);if (y.empty()) return -100;
        const int n=y.w*y.h;
        for (int t=0;t<y.d;++t) {
            std::vector<ncnn::VkMat> qkv(3),result(1);
            for (int a=0;a<3;++a) {
                qkv[a].create(y.c,n,1,size_t(4),1,opt.workspace_vkallocator);if (qkv[a].empty()) return -100;
                map(x[a],qkv[a],{2,n*y.c,y.w,y.h,y.d,y.c,int(x[a].cstep),y.c,n,1,1,int(qkv[a].cstep),0,0,0,0,0,t,0},cmd);
            }
            if (sdpa_->forward(qkv,result,cmd,opt)!=0 || result[0].empty()) return -1;
            map(result[0],y,{3,n*y.c,y.c,n,1,1,int(result[0].cstep),y.w,y.h,y.d,y.c,int(y.cstep),0,0,0,0,0,t,0},cmd);
        }
        return 0;
    }
private:std::unique_ptr<ncnn::Layer> sdpa_;
};
template<class T> ncnn::Layer *create(void *) {return new T;}
}
int register_video_layers(ncnn::Net &net) {
    return net.register_custom_layer("SeedVR2TemporalConv",create<TemporalConv>) ||
        net.register_custom_layer("SeedVR2FrameNorm",create<FrameNorm>) ||
        net.register_custom_layer("SeedVR2FrameSDPA",create<FrameSDPA>) ||
        net.register_custom_layer("SeedVR2TemporalShuffle",create<TemporalShuffle>) ? -1:0;
}
}
