#pragma once
#include "seedvr2/memory.hpp"
#include <optional>

namespace seedvr2::memory {
struct DeviceBudget {
    std::uint64_t budget_bytes = 0, usage_bytes = 0;
    std::uint32_t heap_index = 0;
};
struct WeightDecision {
    bool host = false;
    const char *reason = "explicit_device";
    std::uint64_t estimated_weight_bytes = 0, reserve_bytes = 0;
    std::optional<std::uint64_t> available_bytes;
};
// Pure policy: no model geometry, ncnn, JSON, allocator, global state or threads.
void validate(const MemoryOptions &, bool vulkan);
const char *name(WeightPlacement);
std::uint64_t weight_estimate(std::uint64_t file_bytes);
WeightDecision choose(const MemoryOptions &, std::uint64_t estimated_weight_bytes,
                      std::optional<DeviceBudget>);
} // namespace seedvr2::memory
