#pragma once
#include "seedvr2/image.hpp"
namespace seedvr2::engine {
struct VideoRequest : ImageRequest {
    int max_frames = 17;
};
Result<std::string> inspect_video(const std::filesystem::path &path);
Result<std::string> inspect_video_package(const std::filesystem::path &directory);
Result<std::string> run_video(const VideoRequest &request, ImageObserver observer = {}, CancellationCheck cancelled = {});
}
