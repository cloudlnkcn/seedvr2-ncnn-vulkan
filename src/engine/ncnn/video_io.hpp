#pragma once
#include "image_io.hpp"
#include <functional>
#include <string>
namespace seedvr2::engine::detail {
struct VideoClip {
    std::vector<ImagePixels> frames;
    std::vector<std::int64_t> timestamps; // 90 kHz, relative to the first frame
    int width = 0, height = 0;
    double fps = 0, duration_seconds = 0;
    bool truncated = false, has_audio = false;
};
VideoClip load_video(const std::filesystem::path &, int max_frames, const std::function<bool()> &cancelled = {}, bool probe_only = false);
void save_video(const std::filesystem::path &, const ncnn::Mat &, const VideoClip &,
                const std::function<bool()> &cancelled = {});
std::string media_version();
}
