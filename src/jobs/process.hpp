#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
namespace seedvr2::jobs_detail {
int run_process(const std::vector<std::string> &arguments, const std::filesystem::path &log,
                const std::function<bool()> &cancelled,
                const std::function<void(const std::string &)> &line);
}
