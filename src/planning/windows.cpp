// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026: C++ value types, bounded planning, integer slicing and errors.
// Window semantics derived from ByteDance-Seed/SeedVR models/dit_v2/window.py.
// Pinned source, attribution and Apache-2.0 terms: docs/provenance.md.
#include "seedvr2/planning.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace seedvr2 {
namespace {
std::uint64_t ceil_div(std::uint64_t n, std::uint64_t d) { return n / d + (n % d != 0); }
std::uint32_t ties_to_even(double x) {
    const auto base = static_cast<std::uint32_t>(std::floor(x));
    const auto fraction = x - base;
    return base + (fraction > 0.5 || (fraction == 0.5 && base % 2 != 0));
}
std::vector<Interval> partition(std::uint32_t extent, std::uint32_t size, bool shifted) {
    const bool half_shift = shifted && size < extent;
    const auto count =
        half_shift ? ceil_div(2ULL * extent - 1, 2ULL * size) + 1 : ceil_div(extent, size);
    std::vector<Interval> ranges;
    ranges.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        // Integer division truncates toward zero, matching Python int().
        const auto doubled = 2 * static_cast<std::int64_t>(i) - (half_shift ? 1 : 0);
        const auto begin = std::max<std::int64_t>(0, doubled * size / 2);
        const auto end = std::min<std::int64_t>(extent, (doubled + 2) * size / 2);
        if (begin < end)
            ranges.push_back({static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end)});
    }
    return ranges;
}
} // namespace

Result<WindowPlan> plan_windows(TokenGrid grid, bool shifted) {
    if (grid.t == 0 || grid.h == 0 || grid.w == 0)
        return Error{"INVALID_DIMENSION", "token_grid", "Token dimensions must be positive"};
    // Check before multiplication: even uint64_t can overflow for three uint32_t axes.
    const std::uint64_t area = std::uint64_t{grid.h} * grid.w;
    if (area > max_planning_tokens || grid.t > max_planning_tokens / area)
        return Error{"PLANNING_LIMIT", "token_grid", "Token count exceeds the host planning limit"};
    const double scale = std::sqrt(3600.0 / static_cast<double>(area));
    const auto resized_h = ties_to_even(grid.h * scale);
    const auto resized_w = ties_to_even(grid.w * scale);
    if (resized_h == 0 || resized_w == 0)
        return Error{"UNSUPPORTED_ASPECT", "token_grid",
                     "Official proxy resolution rounds an axis to zero"};
    const TokenGrid nominal{static_cast<std::uint32_t>(ceil_div(std::min(grid.t, 30U), 4)),
                            static_cast<std::uint32_t>(ceil_div(resized_h, 3)),
                            static_cast<std::uint32_t>(ceil_div(resized_w, 3))};
    // An axis partition alone can be too large, before constructing the Cartesian product.
    const auto axis_bound = [&](std::uint32_t n, std::uint32_t size) {
        return ceil_div(n, size) + (shifted ? 1 : 0);
    };
    const auto nt = axis_bound(grid.t, nominal.t);
    const auto nh = axis_bound(grid.h, nominal.h);
    const auto nw = axis_bound(grid.w, nominal.w);
    if (nt > max_planning_windows || nh > max_planning_windows / nt ||
        nw > max_planning_windows / (nt * nh))
        return Error{"PLANNING_LIMIT", "windows",
                     "Window metadata exceeds the host planning limit"};
    const auto ts = partition(grid.t, nominal.t, shifted);
    const auto hs = partition(grid.h, nominal.h, shifted);
    const auto ws = partition(grid.w, nominal.w, shifted);
    WindowPlan plan{grid, shifted, nominal, {}};
    plan.windows.reserve(ts.size() * hs.size() * ws.size());
    std::uint32_t offset = 0;
    for (const auto w : ws)
        for (const auto h : hs)
            for (const auto t : ts) {
                const auto count = (t.end - t.begin) * (h.end - h.begin) * (w.end - w.begin);
                plan.windows.push_back({t, h, w, offset, count});
                offset += count;
            }
    return plan;
}
} // namespace seedvr2
