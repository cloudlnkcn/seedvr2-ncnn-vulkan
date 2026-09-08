#pragma once
#include <span>
#include <string_view>
namespace seedvr2::server {
struct Asset {
    std::string_view path, mime, body;
};
std::span<const Asset> studio_assets();
} // namespace seedvr2::server
