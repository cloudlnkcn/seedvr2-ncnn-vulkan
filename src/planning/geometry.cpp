#include "seedvr2/planning.hpp"

namespace seedvr2 {
namespace {
std::uint64_t ceil_div(std::uint64_t x, std::uint64_t y) { return x / y + (x % y != 0); }
} // namespace

Result<GeometryPlan> plan_geometry(const PlanningRequest &request) {
    const auto in = request.input;
    if (in.width == 0 || in.height == 0 || in.frames == 0)
        return Error{"INVALID_DIMENSION", "media", "Dimensions and frame count must be positive"};
    if (in.width > max_pixel_dimension || in.height > max_pixel_dimension ||
        in.frames > max_pixel_frames)
        return Error{"PLANNING_LIMIT", "media", "Declared input exceeds host planner limits"};
    if (request.kind != MediaKind::image && request.kind != MediaKind::video)
        return Error{"INVALID_KIND", "media.kind", "Unknown media kind"};
    if (request.kind == MediaKind::image && in.frames != 1)
        return Error{"INVALID_FRAME_COUNT", "media.frames", "An image must have exactly one frame"};
    const auto scale = request.scale;
    if (scale.numerator == 0 || scale.denominator == 0 || scale.numerator > 64 ||
        scale.denominator > 64)
        return Error{"INVALID_SCALE", "output.scale",
                     "Scale numerator and denominator must be 1..64"};

    // Round each positive rational dimension to nearest, with ties upward.
    const auto scaled = [&](std::uint32_t dimension) {
        const std::uint64_t n = std::uint64_t{dimension} * scale.numerator;
        return (n + scale.denominator / 2) / scale.denominator;
    };
    const auto width = scaled(in.width);
    const auto height = scaled(in.height);
    if (width == 0 || height == 0 || width > max_pixel_dimension || height > max_pixel_dimension)
        return Error{"PLANNING_LIMIT", "output.scale",
                     "Scaled output is zero or exceeds host planner limits"};

    GeometryPlan plan;
    plan.logical = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                    in.frames};
    plan.working = {static_cast<std::uint32_t>(ceil_div(width, 16) * 16),
                    static_cast<std::uint32_t>(ceil_div(height, 16) * 16),
                    static_cast<std::uint32_t>(ceil_div(in.frames - 1, 4) * 4 + 1)};
    plan.tokens = {(plan.working.frames - 1) / 4 + 1, plan.working.height / 16,
                   plan.working.width / 16};
    plan.token_count = std::uint64_t{plan.tokens.t} * plan.tokens.h * plan.tokens.w;
    if (plan.token_count > max_planning_tokens)
        return Error{"PLANNING_LIMIT", "token_grid", "Token count exceeds the host planning limit"};
    plan.hidden_fp16_bytes = plan.token_count * 2560 * 2;
    return plan;
}
} // namespace seedvr2
