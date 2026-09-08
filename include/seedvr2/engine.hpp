#pragma once
#include "seedvr2/result.hpp"
#include <filesystem>
#include <string>

namespace seedvr2::engine {
struct CaseRequest {
    std::filesystem::path case_file;
    std::filesystem::path output_directory;
    bool vulkan = false;
    int gpu_index = -1;
    int threads = 4;
};
std::string build_status();
Result<std::string> devices();
Result<std::string> run_awa(const CaseRequest &request);
Result<std::string> run_graph(const CaseRequest &request);
Result<std::string> run_dit_block(const CaseRequest &request);
Result<std::string> self_test(bool vulkan, int gpu_index = -1, int threads = 4);
} // namespace seedvr2::engine
