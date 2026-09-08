#include "awa.hpp"
#include "awa_shaders.hpp"
#include "seedvr2/planning.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <command.h>
#include <limits>
#include <memory>
#include <modelbin.h>
#include <pipeline.h>

namespace seedvr2::engine {
namespace {
constexpr int dim = 128;
constexpr int norm_count = 4 * dim;
constexpr int rope_frequencies = 21;
constexpr int rope_positions = 1024; // Pinned official language-RoPE table bound.
constexpr int rope_table_offset = norm_count + rope_frequencies;
constexpr std::uint64_t scratch_limit = 512ULL * 1024 * 1024;

class AdaptiveWindowAttention final : public ncnn::Layer {
  public:
    explicit AdaptiveWindowAttention(ExecutionTrace *trace) : trace_(trace) {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
        support_vulkan = true;
        support_vulkan_packing = false;
        support_vulkan_any_packing = false;
        support_fp16_storage = false;
        support_bf16_storage = false;
    }
    int load_param(const ncnn::ParamDict &pd) override {
        heads_ = pd.get(0, 0);
        shifted_ = pd.get(1, 0);
        epsilon_ = pd.get(2, 1e-5f);
        format_ = pd.get(3, 0);
        if (heads_ < 1 || heads_ > 20 || (shifted_ != 0 && shifted_ != 1) || (format_ != 1 && format_ != 2) ||
            !std::isfinite(epsilon_) || epsilon_ <= 0.f || epsilon_ > 0.01f)
            return -1;
        trace_->heads = heads_;
        trace_->shifted = shifted_;
        return 0;
    }
    int load_model(const ncnn::ModelBin &mb) override {
        if (format_ == 2) {
            const auto epsilon = mb.load(1, 1);
            if (epsilon.empty() || epsilon[0] != epsilon_)
                return -1;
        }
        const auto weights = mb.load(norm_count, 1);
        if (weights.empty())
            return -100;
        norm_.create(rope_table_offset + rope_positions * rope_frequencies * 2, size_t(4));
        if (norm_.empty())
            return -100;
        for (int i = 0; i < norm_count; ++i) {
            const float v = weights[i];
            if (!std::isfinite(v))
                return -1;
            norm_[i] = v;
        }
        for (int i = 0; i < 21; ++i)
            norm_[norm_count + i] = 1.f / std::pow(10000.f, static_cast<float>(2 * i) / 42.f);
        if (format_ == 2) {
            // Preserve pnnx's attribute order in mixed native/custom graphs.
            // This consumes epsilon, norm_weights, rope_frequencies, spec without
            // rewriting or searching through unrelated model-weight bytes.
            const auto frequencies = mb.load(21, 1), spec = mb.load(3, 1);
            if (frequencies.empty() || spec.empty())
                return -1;
            std::array<std::int32_t, 3> values{};
            std::memcpy(values.data(), spec.data, sizeof(values));
            if (values != std::array<std::int32_t, 3>{1, heads_, shifted_})
                return -1;
            for (int i = 0; i < 21; ++i)
                if (!std::isfinite(frequencies[i]) ||
                    std::abs(frequencies[i]-norm_[norm_count+i]) > norm_[norm_count+i]*1e-6f)
                    return -1;
        }
        // Vulkan's fast sin/cos range reduction loses phase accuracy for the
        // text-offset temporal positions. Evaluate the small, input-independent
        // FP32 angle table once on the host; all tensor arithmetic stays on GPU.
        for (int position = 0; position < rope_positions; ++position) {
            for (int pair = 0; pair < rope_frequencies; ++pair) {
                const float angle = static_cast<float>(position) * norm_[norm_count + pair];
                const int offset = rope_table_offset + (position * rope_frequencies + pair) * 2;
                norm_[offset] = std::cos(angle);
                norm_[offset + 1] = std::sin(angle);
            }
        }
        return 0;
    }
    int create_pipeline(const ncnn::Option &opt) override {
        // This baseline is intentionally FP32 at every AWA stage.
        if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_fp16_arithmetic ||
            opt.use_bf16_storage || opt.use_bf16_packed)
            return -1;
        sdpa_cpu_.reset(ncnn::create_layer_cpu("SDPA"));
        scale_cpu_.reset(ncnn::create_layer_cpu("BinaryOp"));
        if (!sdpa_cpu_ || !scale_cpu_)
            return -1;
        ncnn::ParamDict params;
        params.set(6, 1.f); // Q and K are scaled before the dot product, as in FP32-B.
        ncnn::ParamDict scale_params;
        scale_params.set(0, 2); // Multiply by scalar.
        scale_params.set(1, 1);
        scale_params.set(2, static_cast<float>(std::sqrt(1. / std::sqrt(double(dim)))));
        if (sdpa_cpu_->load_param(params) != 0 || sdpa_cpu_->create_pipeline(opt) != 0 ||
            scale_cpu_->load_param(scale_params) != 0 || scale_cpu_->create_pipeline(opt) != 0)
            return -1;
        if (!opt.use_vulkan_compute)
            return 0;
        if (!vkdev)
            return -1;
        sdpa_vk_.reset(ncnn::create_layer_vulkan("SDPA"));
        scale_vk_.reset(ncnn::create_layer_vulkan("BinaryOp"));
        if (!sdpa_vk_ || !sdpa_vk_->support_vulkan || !scale_vk_ || !scale_vk_->support_vulkan)
            return -1;
        sdpa_vk_->vkdev = vkdev;
        scale_vk_->vkdev = vkdev;
        if (sdpa_vk_->load_param(params) != 0 || sdpa_vk_->create_pipeline(opt) != 0 ||
            scale_vk_->load_param(scale_params) != 0 || scale_vk_->create_pipeline(opt) != 0)
            return -1;
        gather_ = std::make_unique<ncnn::Pipeline>(vkdev);
        scatter_ = std::make_unique<ncnn::Pipeline>(vkdev);
        mean_ = std::make_unique<ncnn::Pipeline>(vkdev);
        gather_->set_local_size_xyz(128, 1, 1);
        scatter_->set_local_size_xyz(64, 1, 1);
        mean_->set_local_size_xyz(64, 1, 1);
        if (gather_->create(shaders::awa_gather, sizeof(shaders::awa_gather), {}) != 0 ||
            scatter_->create(shaders::awa_scatter, sizeof(shaders::awa_scatter), {}) != 0 ||
            mean_->create(shaders::awa_text_mean, sizeof(shaders::awa_text_mean), {}) != 0)
            return -1;
        if (gather_->shader_info().push_constant_count != 15 ||
            scatter_->shader_info().push_constant_count != 13 ||
            mean_->shader_info().push_constant_count != 3)
            return -1;
        return 0;
    }
    int destroy_pipeline(const ncnn::Option &opt) override {
        if (sdpa_cpu_)
            sdpa_cpu_->destroy_pipeline(opt);
        if (sdpa_vk_)
            sdpa_vk_->destroy_pipeline(opt);
        if (scale_cpu_)
            scale_cpu_->destroy_pipeline(opt);
        if (scale_vk_)
            scale_vk_->destroy_pipeline(opt);
        sdpa_cpu_.reset();
        sdpa_vk_.reset();
        scale_cpu_.reset();
        scale_vk_.reset();
        gather_.reset();
        scatter_.reset();
        mean_.reset();
        norm_gpu_.release();
        return 0;
    }
    int upload_model(ncnn::VkTransfer &cmd, const ncnn::Option &opt) override {
        if (opt.use_vulkan_compute)
            cmd.record_upload(norm_, norm_gpu_, opt);
        return 0;
    }
    template <typename Mat> bool prepare(const std::vector<Mat> &in, WindowPlan &plan) const {
        if (in.size() != 2 || in[0].n != 1 || in[1].n != 1 || in[0].dims != 4 || in[1].dims != 2 || in[0].elempack != 1 ||
            in[1].elempack != 1 || in[0].elemsize != 4 || in[1].elemsize != 4 ||
            in[0].w != 3 * heads_ * dim || in[1].w != in[0].w || in[0].c <= 0 || in[0].d <= 0 ||
            in[0].h <= 0 || in[0].c > 256 || in[0].d > 128 || in[0].h > 128 || in[1].h <= 0 ||
            in[1].h > 256)
            return false;
        auto result =
            plan_windows({static_cast<std::uint32_t>(in[0].c), static_cast<std::uint32_t>(in[0].d),
                          static_cast<std::uint32_t>(in[0].h)},
                         shifted_ != 0);
        if (is_error(result))
            return false;
        plan = std::get<WindowPlan>(std::move(result));
        if (plan.windows.empty() || plan.windows.size() > 256)
            return false;
        std::uint64_t bytes =
            static_cast<std::uint64_t>(in[0].c) * in[0].d * in[0].h * heads_ * dim * 4;
        bytes += plan.windows.size() * static_cast<std::uint64_t>(in[1].h) * heads_ * dim * 4;
        for (const auto &w : plan.windows) {
            if (w.t.end - w.t.begin + in[1].h > 1024 || w.h.end - w.h.begin > 128 ||
                w.w.end - w.w.begin > 128)
                return false;
            const auto n = w.token_count + static_cast<std::uint64_t>(in[1].h);
            // Conservative lifetime bound for queued FP32 QKV, output, and dense SDPA scratch.
            bytes += heads_ * 4 * (4 * n * dim + n * n);
            if (n > 8192 || bytes > scratch_limit)
                return false;
        }
        return true;
    }
    void normalize_rotate(const float *raw, float *q, float *k, float *v, int head, bool text,
                          std::array<int, 3> pos) const {
        const int hd = heads_ * dim;
        const float *rq = raw + head * dim, *rk = rq + hd, *rv = rk + hd;
        float qs = 0.f, ks = 0.f;
        for (int d = 0; d < dim; ++d) {
            qs += rq[d] * rq[d];
            ks += rk[d] * rk[d];
        }
        qs = 1.f / std::sqrt(qs / dim + epsilon_);
        ks = 1.f / std::sqrt(ks / dim + epsilon_);
        const float *qw = static_cast<const float *>(norm_) + (text ? 256 : 0), *kw = qw + dim;
        for (int d = 0; d < dim; ++d) {
            q[d] = rq[d] * qs * qw[d];
            k[d] = rk[d] * ks * kw[d];
            v[d] = rv[d];
        }
        for (int d = 0; d < 126; d += 2) {
            const float angle = static_cast<float>(pos[d / 42]) * norm_[norm_count + (d % 42) / 2];
            const float c = std::cos(angle), s = std::sin(angle), q0 = q[d], k0 = k[d];
            q[d] = q0 * c - q[d + 1] * s;
            q[d + 1] = q0 * s + q[d + 1] * c;
            k[d] = k0 * c - k[d + 1] * s;
            k[d + 1] = k0 * s + k[d + 1] * c;
        }
    }
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &opt) const override {
        WindowPlan plan;
        if (!prepare(in, plan) || out.size() != 2 || !sdpa_cpu_)
            return -1;
        ++trace_->cpu_calls;
        const auto &vid = in[0], &txt = in[1];
        const int hd = heads_ * dim;
        out[0].create(hd, vid.h, vid.d, vid.c, size_t(4), 1, opt.blob_allocator);
        out[1].create(hd, txt.h, size_t(4), 1, opt.blob_allocator);
        if (out[0].empty() || out[1].empty())
            return -100;
        out[1].fill(0.f);
        for (const auto &win : plan.windows) {
            const int nv = static_cast<int>(win.token_count), n = nv + txt.h;
            std::vector<ncnn::Mat> qkv(3);
            for (auto &x : qkv) {
                x.create(dim, n, heads_, size_t(4), 1, opt.workspace_allocator);
                if (x.empty())
                    return -100;
            }
            const int nh = static_cast<int>(win.h.end - win.h.begin),
                      nw = static_cast<int>(win.w.end - win.w.begin);
#pragma omp parallel for num_threads(opt.num_threads)
            for (int head = 0; head < heads_; ++head) {
                auto q = qkv[0].channel(head), k = qkv[1].channel(head), v = qkv[2].channel(head);
                for (int i = 0; i < n; ++i) {
                    const bool text = i >= nv;
                    const int t = i / (nh * nw), h = (i / nw) % nh, w = i % nw;
                    const float *raw = text ? txt.row(i - nv)
                                            : vid.channel(static_cast<int>(win.t.begin) + t)
                                                  .row((static_cast<int>(win.h.begin) + h) * vid.h +
                                                       static_cast<int>(win.w.begin) + w);
                    const std::array<int, 3> pos = text ? std::array<int, 3>{i - nv, i - nv, i - nv}
                                                        : std::array<int, 3>{txt.h + t, h, w};
                    normalize_rotate(raw, q.row(i), k.row(i), v.row(i), head, text, pos);
                }
            }
            if (trace_->observe_cpu)
                trace_->observe_cpu("awa.qkv", static_cast<int>(trace_->windows.load()), qkv);
            for (int i = 0; i < 2; ++i)
                if (scale_cpu_->forward_inplace(qkv[i], opt) != 0)
                    return -1;
            if (trace_->observe_cpu)
                trace_->observe_cpu("awa.sdpa_inputs", static_cast<int>(trace_->windows.load()), qkv);
            std::vector<ncnn::Mat> result(1);
            const int ret = sdpa_cpu_->forward(qkv, result, opt);
            if (ret != 0)
                return ret;
            if (trace_->observe_cpu)
                trace_->observe_cpu("awa.sdpa", static_cast<int>(trace_->windows.load()), result);
            ++trace_->sdpa_calls;
            ++trace_->windows;
            for (int head = 0; head < heads_; ++head) {
                auto channel = result[0].channel(head);
                for (int i = 0; i < n; ++i) {
                    const float *src = channel.row(i);
                    if (i >= nv) {
                        float *dst = out[1].row(i - nv) + head * dim;
                        for (int d = 0; d < dim; ++d)
                            dst[d] += src[d];
                    } else {
                        const int t = i / (nh * nw), h = (i / nw) % nh, w = i % nw;
                        float *dst = out[0]
                                         .channel(static_cast<int>(win.t.begin) + t)
                                         .row((static_cast<int>(win.h.begin) + h) * vid.h +
                                              static_cast<int>(win.w.begin) + w) +
                                     head * dim;
                        std::copy_n(src, dim, dst);
                    }
                }
            }
        }
        const float count = static_cast<float>(plan.windows.size());
        for (int i = 0; i < txt.h; ++i)
            for (int d = 0; d < hd; ++d)
                out[1].row(i)[d] /= count;
        return 0;
    }
    int forward(const std::vector<ncnn::VkMat> &in, std::vector<ncnn::VkMat> &out,
                ncnn::VkCompute &cmd, const ncnn::Option &opt) const override {
        WindowPlan plan;
        if (!prepare(in, plan) || out.size() != 2 || !sdpa_vk_ || norm_gpu_.empty())
            return -1;
        ++trace_->vulkan_calls;
        const auto &vid = in[0], &txt = in[1];
        const int hd = heads_ * dim;
        out[0].create(hd, vid.h, vid.d, vid.c, size_t(4), 1, opt.blob_vkallocator);
        out[1].create(hd, txt.h, size_t(4), 1, opt.blob_vkallocator);
        ncnn::VkMat text_windows(hd, txt.h, static_cast<int>(plan.windows.size()), size_t(4), 1,
                                 opt.workspace_vkallocator);
        if (out[0].empty() || out[1].empty() || text_windows.empty())
            return -100;
        int window_index = 0;
        for (const auto &win : plan.windows) {
            const int n = static_cast<int>(win.token_count) + txt.h;
            std::vector<ncnn::VkMat> qkv(3);
            for (auto &x : qkv) {
                x.create(dim, n, heads_, size_t(4), 1, opt.workspace_vkallocator);
                if (x.empty())
                    return -100;
            }
            std::vector<ncnn::vk_constant_type> c(15);
            std::array<int, 14> ints = {vid.w,
                                        vid.d,
                                        vid.h,
                                        static_cast<int>(vid.cstep),
                                        txt.h,
                                        heads_,
                                        n,
                                        static_cast<int>(qkv[0].cstep),
                                        static_cast<int>(win.t.begin),
                                        static_cast<int>(win.h.begin),
                                        static_cast<int>(win.w.begin),
                                        static_cast<int>(win.t.end - win.t.begin),
                                        static_cast<int>(win.h.end - win.h.begin),
                                        static_cast<int>(win.w.end - win.w.begin)};
            for (std::size_t i = 0; i < ints.size(); ++i)
                c[i].i = ints[i];
            c[14].f = epsilon_;
            ncnn::Mat dispatch;
            dispatch.w = 128 * n;
            dispatch.h = heads_;
            dispatch.c = 1;
            cmd.record_pipeline(gather_.get(), {vid, txt, norm_gpu_, qkv[0], qkv[1], qkv[2]}, {}, c,
                                dispatch);
            if (trace_->observe_vulkan)
                trace_->observe_vulkan("awa.qkv", window_index, qkv, cmd, opt);
            for (int i = 0; i < 2; ++i)
                if (scale_vk_->forward_inplace(qkv[i], cmd, opt) != 0)
                    return -1;
            if (trace_->observe_vulkan)
                trace_->observe_vulkan("awa.sdpa_inputs", window_index, qkv, cmd, opt);
            std::vector<ncnn::VkMat> result(1);
            const int ret = sdpa_vk_->forward(qkv, result, cmd, opt);
            if (ret != 0)
                return ret;
            if (trace_->observe_vulkan)
                trace_->observe_vulkan("awa.sdpa", window_index, result, cmd, opt);
            ++trace_->sdpa_calls;
            ++trace_->windows;
            std::vector<ncnn::vk_constant_type> s(13);
            const std::array<int, 13> si = {n,
                                            heads_,
                                            static_cast<int>(result[0].cstep),
                                            static_cast<int>(out[0].cstep),
                                            vid.h,
                                            ints[8],
                                            ints[9],
                                            ints[10],
                                            ints[11],
                                            ints[12],
                                            ints[13],
                                            window_index,
                                            static_cast<int>(text_windows.cstep)};
            for (std::size_t i = 0; i < si.size(); ++i)
                s[i].i = si[i];
            dispatch.w = n * hd;
            dispatch.h = 1;
            cmd.record_pipeline(scatter_.get(), {result[0], out[0], text_windows}, {}, s, dispatch);
            ++window_index;
        }
        std::vector<ncnn::vk_constant_type> c(3);
        c[0].i = txt.h * hd;
        c[1].i = window_index;
        c[2].i = static_cast<int>(text_windows.cstep);
        ncnn::Mat dispatch;
        dispatch.w = c[0].i;
        dispatch.h = 1;
        dispatch.c = 1;
        cmd.record_pipeline(mean_.get(), {text_windows, out[1]}, {}, c, dispatch);
        return 0;
    }

  private:
    ExecutionTrace *trace_;
    int heads_ = 0, shifted_ = 0, format_ = 1;
    float epsilon_ = 1e-5f;
    ncnn::Mat norm_;
    ncnn::VkMat norm_gpu_;
    std::unique_ptr<ncnn::Layer> sdpa_cpu_, sdpa_vk_, scale_cpu_, scale_vk_;
    std::unique_ptr<ncnn::Pipeline> gather_, scatter_, mean_;
};
ncnn::Layer *create(void *userdata) {
    return new AdaptiveWindowAttention(static_cast<ExecutionTrace *>(userdata));
}
} // namespace
int register_awa(ncnn::Net &net, ExecutionTrace &trace) {
    return net.register_custom_layer("SeedVR2AWA", create, nullptr, &trace);
}
} // namespace seedvr2::engine
