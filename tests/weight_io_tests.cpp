#include "weight_io.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <chrono>

int main() {
#if defined(__linux__)
    const auto path=std::filesystem::temp_directory_path()/
        ("seedvr2-reader-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {std::ofstream f(path,std::ios::binary);f << "abcdefgh";}
        seedvr2::engine::detail::MappedWeights reader(path);
        std::array<char,4> buffer{};
        const void *reference=nullptr;
        if (reader.read(buffer.data(),4)!=4 || std::string_view(buffer.data(),4)!="abcd" ||
            reader.reference(5,&reference)!=0 || reader.consumed() || reader.reference(4,&reference)!=4 ||
            std::string_view(static_cast<const char *>(reference),4)!="efgh" || !reader.consumed() ||
            reader.read(buffer.data(),1)!=0 || reader.reference(1,&reference)!=0)
            throw std::runtime_error("Mapped reader bound or cursor contract failed");
        std::filesystem::remove(path);return 0;
    } catch (const std::exception &e) {std::filesystem::remove(path);std::cerr << e.what() << '\n';return 1;}
#else
    return 77;
#endif
}
