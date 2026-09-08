#pragma once
#include <filesystem>
namespace seedvr2::server {
int serve(int port, const std::filesystem::path &database,
          const std::filesystem::path &worker, const std::filesystem::path &model, const std::filesystem::path &video_model);
}
