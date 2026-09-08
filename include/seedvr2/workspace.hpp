#pragma once
#include "seedvr2/result.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace seedvr2 {
// SQLite stays behind this boundary. CLI does not link the HTTP framework.
class Workspace {
  public:
    explicit Workspace(const std::filesystem::path &database);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    Result<std::string> save(std::string_view kind, std::string_view status,
                             std::string_view payload);
    Result<std::string> list(std::string_view kind);
    Result<std::string> get(std::string_view id);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::filesystem::path default_database_path();
} // namespace seedvr2
