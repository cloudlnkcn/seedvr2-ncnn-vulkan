#include "seedvr2/path.hpp"
#include "seedvr2/protocol.hpp"
#include "seedvr2/workspace.hpp"
#include "server.hpp"
#include <CLI/CLI.hpp>
#include <array>
#include <iostream>

namespace {
std::filesystem::path default_model(const std::filesystem::path &executable, bool video=false) {
    const auto profile=video?"seedvr2-3b-video-fp32-b-v1":"seedvr2-3b-image-fp32-b-v1";
    const auto development=video?".cache/video-package-fp32":".cache/image-package-fp32";
    const auto user_model = seedvr2::default_database_path().parent_path()/"models"/profile;
    const std::array candidates{
        user_model,
        executable.parent_path()/"models"/profile,
        executable.parent_path().parent_path()/"share/seedvr2/models"/profile,
        executable.parent_path().parent_path().parent_path()/development,
        std::filesystem::current_path()/development};
    for (const auto &path : candidates)
        if (std::filesystem::is_regular_file(path/"manifest.json")) return path;
    return user_model;
}
}

int main(int argc, char **argv) {
    CLI::App cli{"SeedVR2 Studio — loopback Web workspace with an independent CLI"};
    int port = 8877;
    std::string database, worker, model, video_model;
    cli.add_option("--port", port, "Local port; 0 selects an available port")
        ->check(CLI::Range(0, 65535));
    cli.add_option("--database", database, "SQLite workspace path");
    cli.add_option("--model", model, "Complete image model package directory; otherwise discover installed/development package")
        ->envname("SEEDVR2_MODEL_DIR");
    cli.add_option("--worker", worker, "Isolated worker executable (defaults to the adjacent binary)");
    cli.add_option("--video-model",video_model,"Temporal VAE + 32-block video package")->envname("SEEDVR2_VIDEO_MODEL_DIR");
    try {
        cli.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return cli.exit(e);
    }
    try {
        auto executable = std::filesystem::absolute(seedvr2::utf8_path(argv[0]));
#if defined(__linux__)
        executable = std::filesystem::read_symlink("/proc/self/exe");
#endif
        return seedvr2::server::serve(port, database.empty() ? seedvr2::default_database_path()
                                                             : seedvr2::utf8_path(database),
            worker.empty() ? executable.parent_path()/"seedvr2-worker" : seedvr2::utf8_path(worker),
            model.empty() ? default_model(executable) : seedvr2::utf8_path(model),
            video_model.empty()?default_model(executable,true):seedvr2::utf8_path(video_model));
    } catch (const std::exception &e) {
        std::cerr << seedvr2::serialize_error({"WEB_START_FAILED", "", e.what()}) << '\n';
        return 5;
    }
}
