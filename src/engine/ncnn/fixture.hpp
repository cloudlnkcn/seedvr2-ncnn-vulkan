#pragma once
#include <string_view>
#include <vector>
namespace seedvr2::engine {
struct FixtureFile {
    std::string_view path, bytes;
};
const std::vector<FixtureFile> &awa_fixture();
} // namespace seedvr2::engine
