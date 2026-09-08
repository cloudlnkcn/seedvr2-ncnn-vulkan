#include "constant.hpp"
#include <allocator.h>
#include <command.h>
#include <cmath>
#include <memory>

namespace seedvr2::engine {
namespace {
// The pinned upstream MemoryData copies from VkWeightAllocator storage, whose
// buffers lack TRANSFER_SRC. Own a small ordinary Vulkan buffer for constants
// instead. This preserves copy semantics without changing the ncnn dependency.
class Constant final : public ncnn::Layer {
public:
    Constant() {
        one_blob_only = false;
        support_inplace = false;
        support_vulkan = true;
        support_packing = false;
        support_vulkan_packing = false;
    }
    int load_param(const ncnn::ParamDict &pd) override {
        return pd.get(0, 0) == 2560 && pd.get(1, 0) == 0 && pd.get(2, 0) == 0 &&
                       pd.get(11, 0) == 0 && pd.get(21, 1) == 1 ? 0 : -1;
    }
    int load_model(const ncnn::ModelBin &mb) override {
        data_ = mb.load(2560, 1);
        if (data_.empty())
            return -100;
        for (int i = 0; i < 2560; ++i)
            if (!std::isfinite(data_[i]))
                return -1;
        return 0;
    }
    int upload_model(ncnn::VkTransfer &cmd, const ncnn::Option &opt) override {
        allocator_ = std::make_unique<ncnn::VkBlobAllocator>(vkdev, 64 * 1024);
        auto upload = opt;
        upload.blob_vkallocator = allocator_.get();
        cmd.record_upload(data_, gpu_, upload, false);
        return gpu_.empty() ? -100 : 0;
    }
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &opt) const override {
        if (!in.empty() || out.size() != 1)
            return -1;
        out[0] = data_.clone(opt.blob_allocator);
        return out[0].empty() ? -100 : 0;
    }
    int forward(const std::vector<ncnn::VkMat> &in, std::vector<ncnn::VkMat> &out,
                ncnn::VkCompute &cmd, const ncnn::Option &opt) const override {
        if (!in.empty() || out.size() != 1)
            return -1;
        cmd.record_clone(gpu_, out[0], opt);
        return out[0].empty() ? -100 : 0;
    }
private:
    ncnn::Mat data_;
    std::unique_ptr<ncnn::VkBlobAllocator> allocator_;
    ncnn::VkMat gpu_; // Release before its allocator.
};
ncnn::Layer *create_constant(void *) { return new Constant; }
}
int register_constant(ncnn::Net &net) {
    return net.register_custom_layer("SeedVR2Constant", create_constant);
}
}
