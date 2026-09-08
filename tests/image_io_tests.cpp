#include "image_io.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    namespace fs = std::filesystem;
    const fs::path fixture(argv[1]);
    const auto temporary = fs::temp_directory_path()/
        ("seedvr2-jpeg-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        for (const auto *name : {"astronaut-degraded", "grayscale-progressive"}) {
            const auto actual = seedvr2::engine::detail::load_image(fixture/(std::string(name)+".jpg"));
            const auto expected = seedvr2::engine::detail::load_image(fixture/(std::string(name)+"-pillow.png"));
            if (actual.width != expected.width || actual.height != expected.height || actual.rgb != expected.rgb)
                throw std::runtime_error(std::string(name)+" differs from independent Pillow pixels");
        }
        fs::create_directory(temporary);
        std::ifstream source(fixture/"astronaut-degraded.jpg", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(source)), {});
        const auto truncated = temporary/"截断 照片.jpg";
        std::ofstream file(truncated, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()/2));
        file.close();
        for (const auto &path : {truncated, fixture/"unsupported-cmyk.jpg"}) {
            bool rejected = false;
            try { (void)seedvr2::engine::detail::load_image(path); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid or unsupported JPEG was accepted");
        }
        fs::remove_all(temporary);
        std::cout << "PASS: RGB/grayscale progressive JPEG equals Pillow, truncated/CMYK rejected\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (fs::exists(temporary)) fs::remove_all(temporary);
        return 1;
    }
}
