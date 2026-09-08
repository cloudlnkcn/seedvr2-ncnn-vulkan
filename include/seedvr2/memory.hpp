#pragma once
#include <cstdint>

namespace seedvr2 {
enum class WeightPlacement { automatic, device, host };
struct MemoryOptions {
    WeightPlacement weights = WeightPlacement::automatic;
    // Zero selects a policy margin of 1/4 of the queried Vulkan heap budget.
    // This is headroom for other allocations, not a measured workspace peak
    // or a guarantee against allocation failure. Nonzero is an explicit margin.
    std::uint64_t gpu_reserve_bytes = 0;
};
} // namespace seedvr2
