#include "memory_policy.hpp"
#include <limits>
#include <stdexcept>

namespace seedvr2::memory {
const char *name(WeightPlacement value) {
    switch (value) {
        case WeightPlacement::automatic: return "auto";
        case WeightPlacement::device: return "device";
        case WeightPlacement::host: return "host";
    }
    throw std::invalid_argument("Unknown weight placement");
}
void validate(const MemoryOptions &options, bool vulkan) {
    name(options.weights);
    if (!vulkan && (options.weights != WeightPlacement::automatic || options.gpu_reserve_bytes))
        throw std::invalid_argument("Weight placement and GPU reserve require the Vulkan backend");
    if (options.weights != WeightPlacement::automatic && options.gpu_reserve_bytes)
        throw std::invalid_argument("GPU reserve requires automatic weight placement");
}
std::uint64_t weight_estimate(std::uint64_t file_bytes) {
    if (file_bytes > std::numeric_limits<std::uint64_t>::max()/2)
        throw std::overflow_error("Weight payload estimate overflow");
    // Reviewed FP32 graph files: allow two payload copies for preparation.
    // This heuristic excludes activation, workspace and allocator overhead.
    return file_bytes*2;
}
WeightDecision choose(const MemoryOptions &options, std::uint64_t bytes,
                      std::optional<DeviceBudget> budget) {
    validate(options, true);
    WeightDecision result;
    result.estimated_weight_bytes = bytes;
    if (options.weights != WeightPlacement::automatic) {
        result.host = options.weights == WeightPlacement::host;
        result.reason = result.host ? "explicit_host" : "explicit_device";
        return result;
    }
    if (!budget || budget->budget_bytes == 0) {
        // Missing telemetry must not be interpreted as unused/unlimited VRAM.
        result.host = true;
        result.reason = "budget_unavailable";
        result.reserve_bytes = options.gpu_reserve_bytes;
        return result;
    }
    const auto available = budget->usage_bytes < budget->budget_bytes ?
        budget->budget_bytes-budget->usage_bytes : 0;
    result.available_bytes = available;
    result.reserve_bytes = options.gpu_reserve_bytes ? options.gpu_reserve_bytes : budget->budget_bytes/4;
    result.host = available < result.reserve_bytes || available-result.reserve_bytes < bytes;
    result.reason = result.host ? "budget_pressure" : "budget_available";
    return result;
}
} // namespace seedvr2::memory
