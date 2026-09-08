#pragma once
#include "seedvr2/planning.hpp"
#include <string>
#include <string_view>

namespace seedvr2 {
inline constexpr std::size_t max_request_bytes = 1024 * 1024;
[[nodiscard]] Result<PlanningRequest> parse_planning_request(std::string_view text);
// Shared planning use case for the CLI and local HTTP adapter.
[[nodiscard]] Result<std::string> evaluate_planning_request(std::string_view text);
[[nodiscard]] std::string serialize_plan(const PlanningRequest &request,
                                         const GeometryPlan &geometry, const WindowPlan &regular,
                                         const WindowPlan &shifted);
[[nodiscard]] std::string serialize_windows(const WindowPlan &plan);
[[nodiscard]] std::string serialize_error(const Error &error);
} // namespace seedvr2
