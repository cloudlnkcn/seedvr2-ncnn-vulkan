#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <net.h>

namespace seedvr2::engine {
struct ExecutionTrace {
    std::atomic<std::uint64_t> cpu_calls{0}, vulkan_calls{0}, windows{0}, sdpa_calls{0};
    int heads = 0, shifted = -1;
    // Optional developer observers. Empty in application/SDK execution.
    std::function<void(const char *, int, const std::vector<ncnn::Mat> &)> observe_cpu;
    std::function<void(const char *, int, const std::vector<ncnn::VkMat> &,
                       ncnn::VkCompute &, const ncnn::Option &)> observe_vulkan;
};
int register_awa(ncnn::Net &net, ExecutionTrace &trace);
} // namespace seedvr2::engine
