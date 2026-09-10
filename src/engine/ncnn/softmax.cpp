#include "softmax.hpp"
#include "awa_shaders.hpp"
#include <command.h>
#include <layer_shader_type.h>
#include <pipeline.h>
#include <memory>

namespace seedvr2::engine {
namespace {
bool fp32(const ncnn::Option &opt) {
    return opt.use_vulkan_compute && !opt.use_fp16_storage && !opt.use_fp16_packed &&
           !opt.use_fp16_arithmetic && !opt.use_bf16_storage && !opt.use_bf16_packed;
}
bool valid(const ncnn::VkMat &x) {
    return !x.empty() && x.n == 1 && x.dims == 3 && x.elempack == 1 && x.elemsize == 4 &&
           x.w > 0 && x.w <= 4096 && x.h > 0 && x.h <= 4096 && x.c > 0 && x.c <= 20;
}
class FP32Softmax final : public ncnn::Layer {
public:
    FP32Softmax() {
        one_blob_only = support_inplace = support_vulkan = true;
        support_packing = support_vulkan_packing = support_vulkan_any_packing = false;
    }
    int load_param(const ncnn::ParamDict &p) override { return p.get(0, 0) == -1 ? 0 : -1; }
    int create_pipeline(const ncnn::Option &opt) override {
        if (!vkdev || !fp32(opt)) return -1;
        pipeline_ = std::make_unique<ncnn::Pipeline>(vkdev);
        pipeline_->set_local_size_xyz(128, 1, 1);
        return pipeline_->create(shaders::fp32_softmax, sizeof(shaders::fp32_softmax), {});
    }
    int destroy_pipeline(const ncnn::Option &) override { pipeline_.reset(); return 0; }
    int forward_inplace(ncnn::VkMat &x, ncnn::VkCompute &cmd, const ncnn::Option &) const override {
        if (!valid(x)) return -1;
        std::vector<ncnn::vk_constant_type> p(3);
        p[0].i = x.w; p[1].i = x.h; p[2].i = static_cast<int>(x.cstep);
        ncnn::Mat dispatch; dispatch.w = x.h * 128; dispatch.h = x.c; dispatch.c = 1;
        cmd.record_pipeline(pipeline_.get(), {x}, {}, p, dispatch);
        return 0;
    }
private:
    std::unique_ptr<ncnn::Pipeline> pipeline_;
};
class FP32SDPA final : public ncnn::Layer {
public:
    FP32SDPA() {
        one_blob_only = support_inplace = false;
        support_vulkan = true;
        support_packing = support_vulkan_packing = support_vulkan_any_packing = false;
    }
    int load_param(const ncnn::ParamDict &p) override {
        return p.get(5, 0) == 0 && p.get(6, 0.f) == 1.f &&
               p.get(7, 0) == 0 && p.get(18, 0) == 0 ? 0 : -1;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (!vkdev || !fp32(opt)) return -1;
        // Reuse the locked ncnn dense FP32 dot products through its installed
        // shader API. Only the intervening softmax evaluation changes.
        for (int i = 0; i < 2; ++i) {
            auto &pipeline = dots_[i];
            pipeline = std::make_unique<ncnn::Pipeline>(vkdev);
            pipeline->set_local_size_xyz(8, 8, 1);
            std::vector<ncnn::vk_specialization_type> s(13);
            s[1].f = i == 0 ? 0.f : 1.f;
            s[6].i = i == 0 ? 1 : 0;
            if (pipeline->create(ncnn::LayerShaderType::sdpa_cross, opt, s)) return -1;
        }
        softmax_.vkdev = vkdev;
        return softmax_.create_pipeline(opt);
    }
    int destroy_pipeline(const ncnn::Option &opt) override {
        for (auto &pipeline : dots_) pipeline.reset();
        return softmax_.destroy_pipeline(opt);
    }
    int forward(const std::vector<ncnn::VkMat> &in, std::vector<ncnn::VkMat> &out,
                ncnn::VkCompute &cmd, const ncnn::Option &opt) const override {
        if (in.size() != 3 || out.size() != 1) return -1;
        const auto &q = in[0], &k = in[1], &v = in[2];
        if (!valid(q) || !valid(k) || !valid(v) || q.w != k.w || k.h != v.h ||
            q.c != k.c || k.c != v.c) return -1;
        ncnn::VkMat scores(k.h, q.h, q.c, size_t(4), 1, opt.workspace_vkallocator);
        out[0].create(v.w, q.h, q.c, size_t(4), 1, opt.blob_vkallocator);
        if (scores.empty() || out[0].empty()) return -100;
        dot(q, k, scores, true, cmd);
        if (softmax_.forward_inplace(scores, cmd, opt)) return -1;
        dot(scores, v, out[0], false, cmd);
        return 0;
    }
private:
    void dot(const ncnn::VkMat &a, const ncnn::VkMat &b, ncnn::VkMat &y,
             bool transpose, ncnn::VkCompute &cmd) const {
        std::vector<ncnn::vk_constant_type> p(11);
        p[0].f = 1.f; p[1].i = a.h; p[2].i = y.w; p[3].i = a.w; p[4].i = a.c;
        p[5].i = 0; p[6].i = 1;
        p[7].i = static_cast<int>(a.cstep); p[8].i = static_cast<int>(b.cstep);
        p[9].i = static_cast<int>(y.cstep); p[10].i = 0;
        ncnn::Mat dispatch; dispatch.w = (y.w + 3) / 4; dispatch.h = (a.h + 3) / 4; dispatch.c = a.c;
        cmd.record_pipeline(dots_[transpose ? 0 : 1].get(), {a, b, y, ncnn::VkMat()}, {}, p, dispatch);
    }
    std::unique_ptr<ncnn::Pipeline> dots_[2];
    FP32Softmax softmax_;
};
ncnn::Layer *create(void *) { return new FP32Softmax; }
}
int register_fp32_softmax(ncnn::Net &net) {
    return net.opt.use_vulkan_compute ? net.register_custom_layer("Softmax", create) : 0;
}
ncnn::Layer *create_fp32_sdpa() { return new FP32SDPA; }
}
