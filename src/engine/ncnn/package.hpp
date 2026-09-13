#pragma once
#include "graph.hpp"
#include "seedvr2/pipeline.hpp"
#include "seedvr2/image.hpp"
namespace seedvr2::engine::inference {
using namespace detail;
inline constexpr auto profile = "seedvr2-3b-image-fp32-b-v1";
inline constexpr auto video_profile = "seedvr2-3b-video-fp32-b-v1";
struct Cancelled final : std::exception {
    const char *what() const noexcept override { return "Processing cancelled"; }
};
struct GraphFiles { std::filesystem::path param, weights; bool fp16_storage = false; };
struct PackageManifest {
    Json document;
    std::string identity;
    std::uint64_t bytes = 0, largest_graph_bytes = 0;
};
PackageManifest inspect_manifest(const std::filesystem::path &root, bool video);
struct Package {
    Json manifest;
    std::string identity;
    std::map<std::string, GraphFiles> graphs;
    ncnn::Mat text, time;
    explicit Package(const std::filesystem::path &root, const std::function<void()> &check = {},
                     bool video = false, ProgressObserver observer = {});
};
RestoreRequest public_request(const seedvr2::engine::ImageRequest &request, bool video = false, int frames = 17);
} // namespace seedvr2::engine::inference
