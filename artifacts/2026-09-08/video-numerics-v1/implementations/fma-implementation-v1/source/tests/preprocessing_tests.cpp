// Catch the unfused interpolation accumulation that perturbs the model input.
#include "image_io.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        const std::filesystem::path root(argv[1]);
        std::size_t failures = 0, checked = 0;
        double maximum = 0.;
        for (const auto &sizes : {std::array<int, 2>{64, 128}, {128, 64}}) {
            const int source = sizes[0], target = sizes[1];
            const auto name = std::to_string(source) + "-to-" + std::to_string(target);
            seedvr2::engine::detail::ImagePixels input{source, source, {}};
            input.rgb.resize(std::size_t(source) * source * 3);
            std::ifstream pixels(root/(name+".rgb8"), std::ios::binary);
            pixels.read(reinterpret_cast<char *>(input.rgb.data()), static_cast<std::streamsize>(input.rgb.size()));
            if (!pixels || pixels.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Input fixture length differs");
            const auto result = seedvr2::engine::detail::prepare_image(input, target);
            if (result.w != target || result.h != target || result.c != 3)
                throw std::runtime_error("Prepared tensor shape differs");
            std::ifstream expected(root/(name+".f32"), std::ios::binary);
            for (int c = 0; c < 3; ++c) {
                const auto plane = result.channel(c);
                for (int i = 0; i < target * target; ++i) {
                    float reference = 0.f;
                    expected.read(reinterpret_cast<char *>(&reference), sizeof(reference));
                    if (!expected || !std::isfinite(reference))
                        throw std::runtime_error("Invalid reference tensor");
                    const double delta = std::abs(static_cast<double>(plane[i]) - reference);
                    maximum = std::max(maximum, delta);
                    ++checked;
                    failures += !std::isfinite(plane[i]) || delta > 2e-7;
                }
            }
            if (expected.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Reference tensor has trailing data");
        }
        std::cout << "checked=" << checked << " max_abs=" << maximum << " violations=" << failures << '\n';
        return checked == 61440 && failures == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
