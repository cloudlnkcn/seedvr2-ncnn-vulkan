#pragma once
#include <filesystem>
#include <string_view>
namespace seedvr2 {
inline std::filesystem::path utf8_path(std::string_view text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
} // namespace seedvr2
