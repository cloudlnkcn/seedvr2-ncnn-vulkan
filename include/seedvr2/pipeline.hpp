#pragma once
#include "seedvr2/result.hpp"
#include "seedvr2/media.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace seedvr2 {
// File-based SDK for the delivered image and joint short-video pipelines.
// Public types do not depend on ncnn, a JSON library, CLI11 or the Web host.
enum class Backend { cpu, vulkan };
enum class WeightIO { buffered, mapped };
struct RestoreRequest {
    std::filesystem::path model_directory, input_file, output_directory;
    MediaKind kind = MediaKind::image;
    Backend backend = Backend::vulkan;
    int gpu_index = -1, threads = 4, long_side = 256, max_frames = 17;
    std::uint64_t seed = 666;
    bool diagnostic_tensors = false;
    WeightIO weight_io = WeightIO::buffered;
};
struct Progress {
    std::string stage;
    int completed = 0, total = 38;
    double elapsed_ms = 0;
};
using ProgressObserver = std::function<void(const Progress &)>;
using CancellationCheck = std::function<bool()>;
struct Preflight {
    int width = 0, height = 0, maximum_output_frames = 1;
    std::uint64_t package_bytes = 0, largest_graph_bytes = 0, output_reserve_bytes = 0;
    std::string model_manifest_sha256, report_json;
};
struct RunResult {
    std::filesystem::path media_file, report_file;
    int width = 0, height = 0, frames = 1;
    double total_ms = 0;
    std::string report_json;
};
struct PackageInfo {
    std::string manifest_sha256, report_json;
    std::uint64_t bytes = 0;
};
// Preflight authenticates manifest identity and checks paths, sizes, parameters,
// input geometry and the selected device. The run verifies all weight hashes.
Result<Preflight> preflight(const RestoreRequest &request);
Result<RunResult> restore(const RestoreRequest &request, ProgressObserver observer = {},
                          CancellationCheck cancelled = {});
Result<PackageInfo> verify_model(const std::filesystem::path &directory, MediaKind kind,
                                 ProgressObserver observer = {}, CancellationCheck cancelled = {});
// Validates both ends and publishes the new directory only after a full copy.
Result<PackageInfo> copy_model(const std::filesystem::path &source, const std::filesystem::path &destination,
                               MediaKind kind, ProgressObserver observer = {}, CancellationCheck cancelled = {});
std::string build_info();
} // namespace seedvr2
