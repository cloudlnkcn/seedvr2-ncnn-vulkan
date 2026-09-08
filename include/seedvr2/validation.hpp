#pragma once
#include "seedvr2/result.hpp"
#include <filesystem>
#include <string>
#include <string_view>

namespace seedvr2 {
// Audit actual local files. A report is not a model certificate.
Result<std::string> audit_model_bundle(const std::filesystem::path &directory);
std::string model_validation_policy();
std::string model_validation_status();
std::string sha256_text(std::string_view text);
Result<std::string> sha256_file(const std::filesystem::path &path);
} // namespace seedvr2
