#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <vector>

namespace seedvr2::engine {
// FP32-B uses the original na.repeat_concat_idx argsort before averaging.
// Equal text IDs may change window order. Frozen metadata fixtures detect a
// standard-library sort that differs from the locked PyTorch CPU toolchain.
inline std::vector<int> text_pool_order(std::span<const int> lengths, int text_length) {
    if (lengths.empty() || text_length < 1 || text_length > 1024) return {};
    int video_length = 0;
    for (int n : lengths) {
        if (n < 1 || n >= 32768 || video_length + n >= 32768) return {};
        video_length += n;
    }
    const auto size = size_t(video_length) + lengths.size() * size_t(text_length);
    // The official CPU implementation switches to radix sorting at 32768.
    // All currently supported image/video grids are below that boundary.
    if (size >= 32768) return {};
    std::vector<int> keys, owners, permutation(size);
    keys.reserve(size); owners.reserve(size);
    int video = 0;
    for (size_t w = 0; w < lengths.size(); ++w) {
        for (int i = 0; i < lengths[w]; ++i) { keys.push_back(video++); owners.push_back(-1); }
        for (int t = 0; t < text_length; ++t) {
            keys.push_back(video_length + t); owners.push_back(static_cast<int>(w));
        }
    }
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    std::vector<int> result(size - size_t(video_length));
    for (size_t i = 0; i < result.size(); ++i) result[i] = owners[permutation[size_t(video_length) + i]];
    return result;
}

inline float text_pool_mean(const float *values, size_t window_stride, std::span<const int> order) {
    // The supported grids have fewer than 256 windows. Match the official
    // outer sum's 16-value chunks, including their final merge order.
    std::array<float, 4> sums{};
    size_t i = 0;
    for (; i + 16 <= order.size();) {
        for (int j = 0; j < 16; ++j, ++i) sums[0] += values[size_t(order[i]) * window_stride];
        for (size_t level = 1; level < sums.size(); ++level) {
            sums[level] += sums[level - 1]; sums[level - 1] = 0;
            if (i & (size_t(15) << (level * 4))) break;
        }
    }
    for (; i < order.size(); ++i) sums[0] += values[size_t(order[i]) * window_stride];
    for (size_t level = 1; level < sums.size(); ++level) sums[0] += sums[level];
    return sums[0] / static_cast<float>(order.size());
}
}
