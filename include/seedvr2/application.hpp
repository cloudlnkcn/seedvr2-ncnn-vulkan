#pragma once
#include "seedvr2/workspace.hpp"

namespace seedvr2 {
class Application {
  public:
    explicit Application(const std::filesystem::path &database) : workspace_(database) {}
    Result<std::string> save_plan(std::string_view request);
    Result<std::string> audit_and_save(const std::filesystem::path &bundle);
    Result<std::string> self_test_and_save(std::string_view request);
    Result<std::string> records(std::string_view kind) { return workspace_.list(kind); }
    Result<std::string> record(std::string_view id) { return workspace_.get(id); }

  private:
    Workspace workspace_;
};
std::string application_capabilities();
} // namespace seedvr2
