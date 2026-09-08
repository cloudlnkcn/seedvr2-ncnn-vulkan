#pragma once
#include "graph.hpp"
#include "../../runtime/memory_policy.hpp"

namespace seedvr2::engine::detail {
inline std::optional<memory::DeviceBudget> device_budget(const ncnn::VulkanDevice *device) {
    if (!device || !device->info.support_VK_EXT_memory_budget() ||
        !ncnn::vkGetPhysicalDeviceMemoryProperties2KHR) return std::nullopt;
    // Resolve the heap from an actual ncnn blob allocation, rather than guessing
    // the largest heap or treating get_heap_budget() as available memory.
    VulkanAllocators allocators(device);
    ncnn::VkMat probe(1, size_t(4), allocators.blob);
    if (probe.empty()) return std::nullopt;
    const auto &properties = device->info.physicalDeviceMemoryProperties();
    const auto type = probe.data->memory_type_index;
    if (type >= properties.memoryTypeCount) return std::nullopt;
    const auto heap = properties.memoryTypes[type].heapIndex;
    if (heap >= properties.memoryHeapCount ||
        !(properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) return std::nullopt;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2KHR observed{};
    observed.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
    observed.pNext = &budget;
    ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(device->info.physicalDevice(), &observed);
    if (!budget.heapBudget[heap]) return std::nullopt;
    return memory::DeviceBudget{std::min<std::uint64_t>(properties.memoryHeaps[heap].size,
        budget.heapBudget[heap]), budget.heapUsage[heap], heap};
}
inline Json budget_json(const std::optional<memory::DeviceBudget> &value) {
    if (!value) return nullptr;
    return {{"heap_index",value->heap_index},{"budget_bytes",value->budget_bytes},
        {"usage_bytes",value->usage_bytes},{"source","VK_EXT_memory_budget_estimate"}};
}
inline Json configure_memory(ncnn::Net &net, const std::filesystem::path &weights,
                             const MemoryOptions &options,
                             const std::function<std::optional<memory::DeviceBudget>(const ncnn::VulkanDevice *)> &reader = device_budget) {
    memory::validate(options,net.opt.use_vulkan_compute);
    Json report{{"policy",memory::name(options.weights)},{"scope","weight placement before graph loading"},
        {"actual_weight_memory","not_instrumented"},{"activation_offload",false},{"oom_recovery",false}};
    if (!net.opt.use_vulkan_compute) {
        report["requested_memory"]="cpu";report["reason"]="cpu_backend";
        return report;
    }
    const auto before = options.weights == WeightPlacement::automatic ? reader(net.vulkan_device()) : std::nullopt;
    const auto estimate = memory::weight_estimate(std::filesystem::file_size(weights));
    const auto decision = memory::choose(options,estimate,before);
    net.opt.use_weights_in_host_memory = decision.host;
    report["requested_memory"] = decision.host ? "host" : "device";
    report["reason"] = decision.reason;
    report["estimated_weight_bytes"] = estimate;
    report["reserve_bytes"] = decision.reserve_bytes;
    report["reserve_policy"] = options.gpu_reserve_bytes ? "explicit_bytes" : "quarter_heap_budget";
    report["available_bytes"] = decision.available_bytes ? Json(*decision.available_bytes) : Json(nullptr);
    report["before_load"] = budget_json(before);
    return report;
}
inline Json memory_summary(const Json &stages) {
    Json result{{"host_requests",0},{"device_requests",0},{"cpu_graphs",0},{"budget_unavailable",0},
        {"scope","placement requests, not measured allocator residency"},{"cache_enabled",false}};
    for (const auto &stage : stages) {
        const auto &entry=stage.at("memory");
        const auto place=entry.at("requested_memory").get<std::string>();
        const auto key=place=="host"?"host_requests":place=="device"?"device_requests":"cpu_graphs";
        result[key]=result[key].get<int>()+1;
        if (entry.at("reason")=="budget_unavailable") result["budget_unavailable"]=result["budget_unavailable"].get<int>()+1;
    }
    return result;
}
} // namespace seedvr2::engine::detail
