#include "silu.hpp"
#include "awa_shaders.hpp"
#include <command.h>
#include <pipeline.h>
#include <memory>

namespace seedvr2::engine {
namespace {
class DiTSiLU final : public ncnn::Layer {
public:
    DiTSiLU() {
        one_blob_only = true;
        support_inplace = true;
        support_vulkan = true;
        support_packing = support_vulkan_packing = support_vulkan_any_packing = false;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        if (!vkdev || !opt.use_vulkan_compute || opt.use_fp16_storage || opt.use_fp16_packed ||
            opt.use_fp16_arithmetic || opt.use_bf16_storage || opt.use_bf16_packed) return -1;
        pipeline_ = std::make_unique<ncnn::Pipeline>(vkdev);
        pipeline_->set_local_size_xyz(128, 1, 1);
        return pipeline_->create(shaders::dit_silu, sizeof(shaders::dit_silu), {});
    }
    int destroy_pipeline(const ncnn::Option &) override { pipeline_.reset(); return 0; }
    int forward_inplace(ncnn::VkMat &x, ncnn::VkCompute &cmd, const ncnn::Option &) const override {
        if (x.empty() || x.n != 1 || x.dims < 1 || x.dims > 4 || x.elempack != 1 || x.elemsize != 4)
            return -1;
        const auto plane = size_t(x.w) * x.h * x.d;
        const auto size = plane * x.c;
        if (size > 1'000'000'000 || x.cstep > 1'000'000'000) return -1;
        std::vector<ncnn::vk_constant_type> parameters(3);
        parameters[0].i = static_cast<int>(size);
        parameters[1].i = static_cast<int>(plane);
        parameters[2].i = static_cast<int>(x.cstep);
        ncnn::Mat dispatch;
        dispatch.w = static_cast<int>(std::min(size, size_t(65536)));
        dispatch.h = static_cast<int>((size + 65535) / 65536);
        dispatch.c = 1;
        cmd.record_pipeline(pipeline_.get(), {x}, {}, parameters, dispatch);
        return 0;
    }
private:
    std::unique_ptr<ncnn::Pipeline> pipeline_;
};
ncnn::Layer *create(void *) { return new DiTSiLU; }
}
int register_dit_silu(ncnn::Net &net) {
    return net.opt.use_vulkan_compute ? net.register_custom_layer("Swish", create) : 0;
}
}
