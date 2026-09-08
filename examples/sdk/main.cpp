#include <seedvr2/pipeline.hpp>
#include <seedvr2/path.hpp>
#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
    using namespace seedvr2;
    if (argc==1) {
        std::cout << build_info() << '\n';
        // Exercise the installed library's error and cancellation contracts.
        RestoreRequest invalid; invalid.threads=0;
        const auto checked=preflight(invalid);
        const auto stopped=restore({}, {}, [] { return true; });
        return is_error(checked) && std::get<Error>(checked).code=="PREFLIGHT_FAILED" &&
            is_error(stopped) && std::get<Error>(stopped).code=="CANCELLED" ? 0 : 1;
    }
    if (argc!=7) {
        std::cerr << "Usage: seedvr2-sdk-example image|video MODEL INPUT OUTPUT cpu|vulkan SIZE\n";
        return 2;
    }
    RestoreRequest request;
    if (std::string_view(argv[1])!="image" && std::string_view(argv[1])!="video") return 2;
    if (std::string_view(argv[5])!="cpu" && std::string_view(argv[5])!="vulkan") return 2;
    request.kind=std::string_view(argv[1])=="video"?MediaKind::video:MediaKind::image;
    request.model_directory=utf8_path(argv[2]); request.input_file=utf8_path(argv[3]); request.output_directory=utf8_path(argv[4]);
    request.backend=std::string_view(argv[5])=="cpu"?Backend::cpu:Backend::vulkan;
    try {request.long_side=std::stoi(argv[6]);} catch (...) {return 2;}
    request.diagnostic_tensors=true;
    const auto result=restore(request,[](const Progress &p) {
        std::cerr << p.stage << ' ' << p.completed << '/' << p.total << '\n';
    });
    if (is_error(result)) {
        const auto &e=std::get<Error>(result); std::cerr << e.code << ": " << e.message << '\n'; return 1;
    }
    const auto &run=std::get<RunResult>(result);
    std::cout << run.report_json << '\n';
    return std::filesystem::is_regular_file(run.media_file) && std::filesystem::is_regular_file(run.report_file)?0:1;
}
