#pragma once
#include "seedvr2/result.hpp"
#include <filesystem>
#include <memory>
#include <string_view>

namespace seedvr2 {
// Local job application service. Its process, database and HTTP adapters are
// private; a caller exchanges versioned JSON and workspace resource identities.
class Jobs {
public:
    Jobs(const std::filesystem::path &database, const std::filesystem::path &worker,
         const std::filesystem::path &model, const std::filesystem::path &video_model = {});
    ~Jobs();
    Jobs(const Jobs &) = delete;
    Jobs &operator=(const Jobs &) = delete;
    Result<std::string> import_media(std::string_view bytes, std::string_view name, bool video = false);
    Result<std::string> submit(std::string_view request);
    Result<std::string> list();
    Result<std::string> get(std::string_view id);
    Result<std::string> cancel(std::string_view id);
    Result<std::string> events(std::string_view id, std::uint64_t after);
    Result<std::filesystem::path> media_file(std::string_view id);
    Result<std::filesystem::path> result_file(std::string_view id, std::string_view name);
    std::string model_status();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace seedvr2
