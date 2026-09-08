#include "seedvr2/image.hpp"
#include "seedvr2/video.hpp"
#include "seedvr2/path.hpp"
#include <CLI/CLI.hpp>
#include <csignal>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#if defined(__linux__)
#include <sys/prctl.h>
#include <unistd.h>
#endif

namespace {
volatile std::sig_atomic_t cancel_requested = 0;
void request_cancel(int) { cancel_requested = 1; }
using Json = nlohmann::json;
void emit(Json value) { value["protocol"] = "seedvr2-worker-v1"; std::cout << value.dump() << std::endl; }
}
int main(int argc, char **argv) {
    CLI::App cli{"SeedVR2 isolated image worker"};
    bool capabilities = false;
    std::string request;
    int parent_pid = 0;
    cli.add_flag("--capabilities", capabilities);
    cli.add_option("--request", request, "Private job request file");
    cli.add_option("--parent-pid", parent_pid);
    try { cli.parse(argc, argv); }
    catch (const CLI::ParseError &e) { return cli.exit(e); }
    if (capabilities) {
        emit({{"type", "capabilities"}, {"inference", true}, {"accepts_run", true},
              {"model_verified", false}, {"image", true}, {"video", true}, {"max_video_frames",17}});
        return 0;
    }
    std::signal(SIGTERM, request_cancel);
    std::signal(SIGINT, request_cancel);
#if defined(__linux__)
    if (parent_pid > 0) {
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent_pid)
            return 125;
    }
#endif
    try {
        const auto path = seedvr2::utf8_path(request);
        if (request.empty() || !std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > 65536)
            throw std::runtime_error("Missing or oversized worker request");
        std::ifstream input(path);
        const auto doc = Json::parse(input);
        if (doc.at("schema_version") != "seedvr2-worker-request-v1" ||
            (doc.at("backend") != "cpu" && doc.at("backend") != "vulkan"))
            throw std::runtime_error("Unsupported worker request");
        const seedvr2::engine::ImageRequest settings{seedvr2::utf8_path(doc.at("model").get<std::string>()),
            seedvr2::utf8_path(doc.at("input").get<std::string>()), seedvr2::utf8_path(doc.at("output").get<std::string>()),
            doc.at("backend") == "vulkan", doc.at("gpu").get<int>(), doc.at("threads").get<int>(),
            doc.at("size").get<int>(), doc.at("seed").get<std::uint64_t>(), false};
        const auto observe=[](const seedvr2::engine::ImageProgress &p) {
            emit({{"type", "progress"}, {"stage", p.stage}, {"completed", p.completed},
                  {"total", p.total}, {"elapsed_ms", p.elapsed_ms}});
        };
        const auto cancelled=[] { return cancel_requested != 0; };
        const auto kind=doc.value("kind","image");
        if (kind!="image" && kind!="video") throw std::runtime_error("Unknown media kind");
        const seedvr2::RestoreRequest native{settings.model_directory,settings.input_file,settings.output_directory,
            kind=="video"?seedvr2::MediaKind::video:seedvr2::MediaKind::image,
            settings.vulkan?seedvr2::Backend::vulkan:seedvr2::Backend::cpu,settings.gpu_index,settings.threads,
            settings.long_side,kind=="video"?doc.at("max_frames").get<int>():1,settings.seed,false};
        const auto result=seedvr2::restore(native,observe,cancelled);
        if (seedvr2::is_error(result)) {
            const auto &error = std::get<seedvr2::Error>(result);
            emit({{"type", "error"}, {"code", error.code}, {"message", error.message}});
            return error.code == "CANCELLED" ? 130 : 1;
        }
        emit({{"type", "result"}, {"report", Json::parse(std::get<seedvr2::RunResult>(result).report_json)}});
        return 0;
    } catch (const std::exception &e) {
        emit({{"type", "error"}, {"code", "WORKER_ERROR"}, {"message", e.what()}});
        return 1;
    }
}
