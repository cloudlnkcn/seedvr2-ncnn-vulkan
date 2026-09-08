#pragma once
#include "seedvr2/media.hpp"

#include "seedvr2/result.hpp"
#include <cstdint>
#include <vector>

namespace seedvr2 {

struct Extent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t frames = 0;
    bool operator==(const Extent &) const = default;
};
struct Ratio {
    std::uint32_t numerator = 1;
    std::uint32_t denominator = 1;
};
// Declared dimensions for planning only. This is not a verified JobSpec.
struct PlanningRequest {
    MediaKind kind = MediaKind::image;
    Extent input;
    Ratio scale;
};
struct TokenGrid {
    std::uint32_t t = 0;
    std::uint32_t h = 0;
    std::uint32_t w = 0;
    bool operator==(const TokenGrid &) const = default;
};
struct GeometryPlan {
    Extent logical;
    Extent working;
    TokenGrid tokens;
    std::uint64_t token_count = 0;
    std::uint64_t hidden_fp16_bytes = 0;
};
struct Interval {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    bool operator==(const Interval &) const = default;
};
struct Window {
    Interval t;
    Interval h;
    Interval w;
    std::uint32_t packed_offset = 0;
    std::uint32_t token_count = 0;
};
struct WindowPlan {
    TokenGrid grid;
    bool shifted = false;
    TokenGrid nominal_window;
    std::vector<Window> windows;
};

// Host safety limits for the prototype planner, not certified model capabilities.
inline constexpr std::uint32_t max_pixel_dimension = 65536;
inline constexpr std::uint32_t max_pixel_frames = 4097;
inline constexpr std::uint64_t max_planning_tokens = 100000000;
inline constexpr std::uint64_t max_planning_windows = 1000000;

[[nodiscard]] Result<GeometryPlan> plan_geometry(const PlanningRequest &request);
[[nodiscard]] Result<WindowPlan> plan_windows(TokenGrid grid, bool shifted);
} // namespace seedvr2
